"""Comprehensive audit of the Smart Sentinel AI trading pipeline on historical XAUUSD data.

Produces a detailed markdown report answering:
  - What is the *achievable* score distribution (news/volume bypassed)?
  - At what threshold does the strategy start trading?
  - What is the edge (win-rate, expectancy, PF) at each threshold?
  - Does edge deteriorate over time (is it a 2020-only phenomenon)?
  - Which exit reasons dominate (tuning signal)?
  - LONG vs SHORT symmetry?
  - Per-year performance?
  - Per-hour-of-day performance?
  - Per-tier (PREMIUM/STANDARD/WEAK) performance?

Outputs:
  - reports/audit/audit_report.md  — full prospect-ready writeup
  - reports/audit/sweep_results.json — raw multi-threshold data
  - reports/audit/trades_combined.csv — trades from the "best" run

Usage:
  python -m scripts.audit_backtest
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_performance, tier_from_score
from src.backtest.news_replay import BacktestNewsProvider
from src.backtest.state_machine_replay import (
    ReplayResults,
    SignalReplay,
    TradeRecord,
    classify_regime_series,
    classify_vol_regime_series,
)
from src.environment.strategy_features import SmartMoneyEngine
from src.intelligence.confluence_detector import ConfluenceDetector
from src.intelligence.signal_state_machine import (
    BarInput,
    SignalStateMachine,
    StateMachineConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

CSV_PATH = Path("data/XAU_15MIN_2019_2026.csv")
CALENDAR_CSV = Path("data/economic_calendar_HIGH_IMPACT_2019_2025.csv")
OUT_DIR = Path("reports/audit")
SYMBOL = "XAUUSD"
TIMEFRAME = "M15"
WARMUP = 100
# Override via env var for ad-hoc experiments: CALENDAR_PATH=... python -m scripts.audit_backtest
import os
_CAL_OVERRIDE = os.environ.get("CALENDAR_PATH")
if _CAL_OVERRIDE:
    CALENDAR_CSV = Path(_CAL_OVERRIDE)

# Production-style configs we want to compare
# Score dist showed max observed = 55.5 across 7 yrs → configs ≥60 are
# structurally 0-trade (kept production_default as the headline demo).
SWEEP_CONFIGS = [
    # (label, enter, exit, confirm, cooldown, max_age)
    ("production_default",  75, 55, 2, 2, 12),
    ("relaxed_55",          55, 40, 2, 2, 12),
    ("relaxed_50",          50, 35, 2, 2, 12),
    ("relaxed_45",          45, 30, 2, 2, 12),
    ("relaxed_40",          40, 25, 2, 2, 12),
    ("relaxed_35",          35, 20, 2, 2, 12),
    ("relaxed_30",          30, 15, 2, 2, 12),
]


# =============================================================================
# DATA LOAD
# =============================================================================

def load_and_enrich(csv_path: Path) -> pd.DataFrame:
    logger.info("Loading %s", csv_path)
    df = pd.read_csv(csv_path)
    for cand in ("timestamp", "Date", "date", "datetime", "time"):
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand])
            df = df.set_index(cand)
            break
    rename: Dict[str, str] = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("open", "high", "low", "close", "volume"):
            rename[col] = lc.capitalize()
    if rename:
        df = df.rename(columns=rename)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    logger.info(
        "Loaded %d bars (%s -> %s)",
        len(df), df.index[0], df.index[-1],
    )
    logger.info("Running SmartMoneyEngine...")
    t0 = time.time()
    engine = SmartMoneyEngine(data=df, config={}, verbose=False)
    enriched = engine.analyze()
    logger.info("Enrichment: %d bars x %d cols in %.1fs",
                len(enriched), len(enriched.columns), time.time() - t0)
    return enriched


# =============================================================================
# SCORE DISTRIBUTION CAPTURE (detector with min_score=0)
# =============================================================================

@dataclass
class ScoreSample:
    ts: str
    score: float
    direction: str  # "LONG" | "SHORT"


def capture_full_score_distribution(
    enriched: pd.DataFrame,
    warmup: int = WARMUP,
    news_provider: Optional[BacktestNewsProvider] = None,
) -> List[ScoreSample]:
    """Run the detector with min_score=0 to capture *every* scored bar.

    This tells us the real achievable score range in offline replay — where
    News (20pts) and Volume (10pts) are structurally at zero.
    """
    from src.backtest.state_machine_replay import _regime_for

    logger.info("Capturing full score distribution (detector.min_score=0)...")
    det = ConfluenceDetector(symbol=SYMBOL, min_score=0.0)
    regime_tags = classify_regime_series(enriched)
    close = enriched["close"] if "close" in enriched.columns else enriched["Close"]
    slow = close.rolling(50, min_periods=50).mean()
    fast = close.rolling(20, min_periods=20).mean()
    slope_bps = (fast - slow) / slow * 10000.0
    vol_tags = classify_vol_regime_series(enriched)

    samples: List[ScoreSample] = []
    for i in range(warmup, len(enriched)):
        row = enriched.iloc[i]
        close_v = float(row.get("close", row.get("Close", 0.0)))
        high_v = float(row.get("high", row.get("High", close_v)))
        low_v = float(row.get("low", row.get("Low", close_v)))
        atr = float(row.get("ATR", 0.0) or 0.0)
        if atr <= 0 or not np.isfinite(atr):
            continue
        if not (0 < low_v <= close_v <= high_v):
            continue

        smc = {
            "BOS_SIGNAL": float(row.get("BOS_SIGNAL", 0) or 0),
            "BOS_EVENT": float(row.get("BOS_EVENT", 0) or 0),
            "FVG_SIGNAL": float(row.get("FVG_SIGNAL", 0) or 0),
            "OB_STRENGTH_NORM": float(row.get("OB_STRENGTH_NORM", 0) or 0),
            "RSI": float(row.get("RSI", 50) or 50),
            "MACD_Diff": float(row.get("MACD_Diff", 0) or 0),
            "CHOCH_SIGNAL": float(row.get("CHOCH_SIGNAL", 0) or 0),
            "CHOCH_DIVERGENCE": float(row.get("CHOCH_DIVERGENCE", 0) or 0),
            "FVG_SIZE_NORM": float(row.get("FVG_SIZE_NORM", 0) or 0),
        }
        regime = _regime_for(regime_tags.iloc[i], float(slope_bps.iloc[i] or 0))
        bar_ts = str(enriched.index[i])
        news = news_provider(bar_ts) if news_provider is not None else None
        sig = det.analyze(
            smc_features=smc, regime=regime, news=news,
            price=close_v, atr=atr,
            volume=None, volume_ma=None,
            bar_timestamp=bar_ts,
            vol_forecast=None,
        )
        if sig is not None:
            samples.append(ScoreSample(
                ts=str(enriched.index[i]),
                score=sig.confluence_score,
                direction=sig.signal_type.value,
            ))
    logger.info("Captured %d scored bars", len(samples))
    return samples


# =============================================================================
# MULTI-THRESHOLD SWEEP
# =============================================================================

def run_sweep(
    enriched: pd.DataFrame,
    news_provider: Optional[BacktestNewsProvider] = None,
) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    for label, enter, exit_th, confirm, cooldown, max_age in SWEEP_CONFIGS:
        logger.info(
            "RUN [%s]  enter=%d exit=%d confirm=%d cooldown=%d max_age=%d",
            label, enter, exit_th, confirm, cooldown, max_age,
        )
        cfg = StateMachineConfig(
            symbol=SYMBOL,
            enter_threshold=float(enter),
            exit_threshold=float(exit_th),
            confirm_bars=confirm,
            cooldown_bars=cooldown,
            max_signal_age_bars=max_age,
        )
        replay = SignalReplay(
            symbol=SYMBOL, timeframe=TIMEFRAME,
            state_machine_config=cfg,
            warmup_bars=WARMUP,
            news_provider=news_provider,
        )
        t0 = time.time()
        res = replay.run(enriched)
        elapsed = time.time() - t0
        metrics = compute_performance(
            res.trades, timeframe=TIMEFRAME,
            tier_fn=lambda t: tier_from_score(t.confluence_score),
            bars_processed=res.bars_processed,
        )
        logger.info(
            "[%s] %d trades, win=%.1f%%, PF=%.2f, total=%.1fR (%.1fs)",
            label, res.total_trades, res.win_rate * 100,
            res.profit_factor, res.total_r, elapsed,
        )
        results[label] = {
            "enter": enter, "exit": exit_th,
            "results": res, "metrics": metrics,
        }
    return results


# =============================================================================
# DEEP ANALYSIS OF A SINGLE RUN
# =============================================================================

def analyse_trades(trades: List[TradeRecord]) -> Dict[str, object]:
    """Stratifications the top-level report alone doesn't give you."""
    if not trades:
        return {"empty": True}

    # By direction
    by_dir: Dict[str, List[TradeRecord]] = {"LONG": [], "SHORT": []}
    for t in trades:
        by_dir.setdefault(t.direction, []).append(t)

    def _stats(ts: List[TradeRecord]) -> Dict[str, float]:
        if not ts:
            return {"n": 0}
        r = [t.r_multiple for t in ts]
        wins = [x for x in r if x > 0]
        losses = [x for x in r if x < 0]
        gw = sum(wins) if wins else 0.0
        gl = sum(losses) if losses else 0.0
        pf = (gw / abs(gl)) if gl < 0 else (float("inf") if gw > 0 else 0.0)
        return {
            "n": len(ts),
            "win_rate": len(wins) / len(ts),
            "expectancy_r": float(np.mean(r)),
            "total_r": float(sum(r)),
            "profit_factor": pf,
            "avg_bars_held": float(np.mean([t.bars_held for t in ts])),
        }

    dir_stats = {d: _stats(ts) for d, ts in by_dir.items()}

    # Per-year
    per_year: Dict[int, List[TradeRecord]] = defaultdict(list)
    for t in trades:
        try:
            y = pd.to_datetime(t.entry_bar).year
            per_year[y].append(t)
        except Exception:
            continue
    year_stats = {int(y): _stats(ts) for y, ts in sorted(per_year.items())}

    # Per-hour
    per_hour: Dict[int, List[TradeRecord]] = defaultdict(list)
    for t in trades:
        try:
            h = pd.to_datetime(t.entry_bar).hour
            per_hour[h].append(t)
        except Exception:
            continue
    hour_stats = {int(h): _stats(ts) for h, ts in sorted(per_hour.items())}

    # Per-tier
    per_tier: Dict[str, List[TradeRecord]] = defaultdict(list)
    for t in trades:
        per_tier[tier_from_score(t.confluence_score)].append(t)
    tier_stats = {k: _stats(ts) for k, ts in per_tier.items()}

    # Holding time distribution
    holds = [t.bars_held for t in trades]

    # Dollar P&L (price-space) per direction
    pnl_long = sum(t.pnl_price for t in by_dir.get("LONG", []))
    pnl_short = sum(t.pnl_price for t in by_dir.get("SHORT", []))

    return {
        "empty": False,
        "by_direction": dir_stats,
        "by_year": year_stats,
        "by_hour": hour_stats,
        "by_tier": tier_stats,
        "holds": {
            "mean": float(np.mean(holds)),
            "median": float(np.median(holds)),
            "p10": float(np.percentile(holds, 10)),
            "p90": float(np.percentile(holds, 90)),
        },
        "pnl_price": {
            "long_usd": pnl_long,
            "short_usd": pnl_short,
            "total_usd": pnl_long + pnl_short,
        },
    }


# =============================================================================
# REPORT RENDERING
# =============================================================================

def render_markdown(
    sweep: Dict[str, dict],
    scores: List[ScoreSample],
    date_range: Tuple[str, str],
    bars_total: int,
) -> str:
    lines: List[str] = []

    # ----- Header -----
    lines += [
        "# Smart Sentinel AI — Audit Backtest Complet XAU/USD",
        "",
        f"**Symbole** : {SYMBOL}  ",
        f"**Timeframe** : {TIMEFRAME}  ",
        f"**Fenêtre** : {date_range[0]} → {date_range[1]}  ",
        f"**Bars totales** : {bars_total:,}  ",
        "",
        "Objectif : identifier les blocages à la commercialisation et "
        "cartographier le comportement de la stratégie à tous les seuils.",
        "",
        "---",
        "",
    ]

    # ----- Score distribution -----
    lines += [
        "## 1. Distribution des scores de confluence",
        "",
        "Le détecteur a tourné avec `min_score=0` pour capturer **tous** les "
        "scores (pas seulement ceux au-dessus du seuil).",
        "",
    ]
    if scores:
        arr = np.array([s.score for s in scores])
        long_arr = np.array([s.score for s in scores if s.direction == "LONG"])
        short_arr = np.array([s.score for s in scores if s.direction == "SHORT"])
        pcts = [10, 25, 50, 75, 90, 95, 99]
        lines.append("| Stat | Tous | LONG | SHORT |")
        lines.append("|---|---:|---:|---:|")
        lines.append(
            f"| Nombre de bars scorées | {len(arr):,} | {len(long_arr):,} | {len(short_arr):,} |"
        )
        lines.append(
            f"| Score min | {arr.min():.1f} | "
            f"{long_arr.min() if len(long_arr) else float('nan'):.1f} | "
            f"{short_arr.min() if len(short_arr) else float('nan'):.1f} |"
        )
        lines.append(
            f"| Score max | {arr.max():.1f} | "
            f"{long_arr.max() if len(long_arr) else float('nan'):.1f} | "
            f"{short_arr.max() if len(short_arr) else float('nan'):.1f} |"
        )
        lines.append(
            f"| Moyenne | {arr.mean():.1f} | "
            f"{long_arr.mean() if len(long_arr) else float('nan'):.1f} | "
            f"{short_arr.mean() if len(short_arr) else float('nan'):.1f} |"
        )
        for p in pcts:
            lines.append(
                f"| P{p} | {np.percentile(arr, p):.1f} | "
                f"{np.percentile(long_arr, p) if len(long_arr) else float('nan'):.1f} | "
                f"{np.percentile(short_arr, p) if len(short_arr) else float('nan'):.1f} |"
            )
        lines.append("")

        # Threshold reachability
        lines += [
            "### Fraction de bars au-dessus de chaque seuil",
            "",
            "| Seuil | Bars ≥ seuil | % de bars scorées |",
            "|---:|---:|---:|",
        ]
        for th in (40, 45, 50, 55, 60, 65, 70, 75, 80):
            n = int((arr >= th).sum())
            pct = 100.0 * n / len(arr)
            lines.append(f"| {th} | {n:,} | {pct:.3f}% |")
        lines.append("")
    else:
        lines.append("_Aucun bar scoré. Détecteur ne produit rien._\n")

    lines += [
        "### Interprétation",
        "",
        "Le ConfluenceDetector utilise 8 composants totalisant 100 points :",
        "- BOS 15  |  FVG 15  |  OrderBlock 10  |  Régime 25  |  **News 20**  |  **Volume 10**  |  Momentum 3  |  RSI div 2",
        "",
        "**En backtest, News et Volume sont structurellement à 0** (pas de flux "
        "historique news, pas de `volume_ma`). Score max théorique atteignable = **70/100**.",
        "",
        "→ Le seuil de production `enter=75` est **mathématiquement impossible** "
        "à atteindre en backtest, et même en live il exige que News+Volume "
        "contribuent quasi-parfaitement à chaque bar — ce qui n'arrive jamais.",
        "",
        "---",
        "",
    ]

    # ----- Sweep summary table -----
    lines += [
        "## 2. Sweep multi-seuils (la vraie mesure de l'edge)",
        "",
        "| Config | enter/exit | Trades | Win% | Expectancy R | Total R | "
        "PF | Max DD R | Sharpe ann. | Trades/an |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, payload in sweep.items():
        res: ReplayResults = payload["results"]
        m = payload["metrics"]
        if m.total_trades == 0:
            lines.append(
                f"| {label} | {payload['enter']}/{payload['exit']} | 0 | — | — | — | — | — | — | — |"
            )
            continue
        sharpe_ann = (f"{m.sharpe_annualised:+.2f}"
                      if m.sharpe_annualised is not None else "—")
        tpy = f"{m.trades_per_year:.0f}" if m.trades_per_year is not None else "—"
        pf_str = (f"{m.profit_factor:.2f}"
                  if np.isfinite(m.profit_factor) else "∞")
        lines.append(
            f"| {label} | {payload['enter']}/{payload['exit']} | "
            f"{m.total_trades} | {m.win_rate*100:.1f}% | "
            f"{m.expectancy_r:+.3f} | {m.total_r:+.2f} | "
            f"{pf_str} | {m.max_drawdown_r:.2f} | "
            f"{sharpe_ann} | {tpy} |"
        )
    lines.append("")

    # Interpretation of sweep
    traded_configs = [
        (label, p) for label, p in sweep.items()
        if p["metrics"].total_trades > 0
    ]
    lines += [
        "### Lecture",
        "",
        "- **Profit factor** : < 1 = stratégie perd de l'argent. >= 1.3 minimum "
        "pour un produit payant. >= 1.5 pour rassurer un acheteur.",
        "- **Expectancy R** : gain moyen (en R = risque initial) par trade. "
        "Doit être positif.",
        "- **Sharpe annualisé** : > 1 correct, > 2 excellent. < 0.5 = "
        "inexploitable commercialement.",
        "",
    ]

    # ----- Deep dive on the best-trading config -----
    best_label, best_payload = None, None
    best_pf = -1.0
    for label, p in sweep.items():
        m = p["metrics"]
        if m.total_trades >= 30 and np.isfinite(m.profit_factor) and m.profit_factor > best_pf:
            best_pf = m.profit_factor
            best_label, best_payload = label, p

    # Fallback: highest-volume traded config
    if best_label is None and traded_configs:
        best_label, best_payload = max(
            traded_configs, key=lambda kv: kv[1]["metrics"].total_trades,
        )

    if best_payload:
        res: ReplayResults = best_payload["results"]
        m = best_payload["metrics"]
        trades = res.trades
        deep = analyse_trades(trades)

        lines += [
            "---",
            "",
            f"## 3. Analyse détaillée — configuration `{best_label}` "
            f"(enter={best_payload['enter']}, exit={best_payload['exit']})",
            "",
            "Cette config est sélectionnée car elle a le meilleur PF parmi celles "
            "ayant au moins 30 trades (échantillon statistiquement pertinent).",
            "",
            "### 3.1 Résumé",
            "",
            "| Métrique | Valeur |",
            "|---|---:|",
            f"| Trades | {m.total_trades} |",
            f"| Wins | {m.wins} ({m.win_rate*100:.1f}%) |",
            f"| Losses | {m.losses} ({m.loss_rate*100:.1f}%) |",
            f"| Breakeven | {m.breakeven} |",
            f"| Expectancy / trade | **{m.expectancy_r:+.3f} R** |",
            f"| Total R | {m.total_r:+.2f} R |",
            f"| Best / Worst | {m.best_r:+.2f} R / {m.worst_r:+.2f} R |",
            f"| Profit factor | **{m.profit_factor:.2f}** |",
            f"| Payoff ratio (avgWin/avgLoss) | {m.payoff_ratio:.2f} |",
            f"| Max drawdown | {m.max_drawdown_r:.2f} R |",
            f"| Max consec losses | {m.max_consecutive_losses} |",
            f"| Sharpe per trade / annualised | {m.sharpe_per_trade:+.2f} / "
            f"{m.sharpe_annualised if m.sharpe_annualised is not None else float('nan'):+.2f} |",
            f"| Sortino annualised | "
            f"{m.sortino_annualised if m.sortino_annualised is not None else float('nan'):+.2f} |",
            f"| Calmar | {'∞' if not np.isfinite(m.calmar) else f'{m.calmar:.2f}'} |",
            "",
        ]

        if not deep["empty"]:
            # Direction
            lines += [
                "### 3.2 LONG vs SHORT",
                "",
                "| Direction | Trades | Win% | Expectancy R | Total R | PF |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for d, st in deep["by_direction"].items():
                if st["n"] == 0:
                    lines.append(f"| {d} | 0 | — | — | — | — |")
                    continue
                pfs = ("∞" if not np.isfinite(st["profit_factor"])
                       else f"{st['profit_factor']:.2f}")
                lines.append(
                    f"| {d} | {st['n']} | {st['win_rate']*100:.1f}% | "
                    f"{st['expectancy_r']:+.3f} | {st['total_r']:+.2f} | {pfs} |"
                )
            lines.append("")

            pnl = deep["pnl_price"]
            lines += [
                "**P&L en $ (price-space, 1 unit de gold par trade)** :",
                "",
                f"- LONG  : ${pnl['long_usd']:+.2f}",
                f"- SHORT : ${pnl['short_usd']:+.2f}",
                f"- **Total** : ${pnl['total_usd']:+.2f}",
                "",
                "_Note : en ajoutant un sizing fixe (ex: 0.1 lot), multiplier par "
                "10. Ne comprend pas spread/commissions._",
                "",
            ]

            # Per-year
            lines += [
                "### 3.3 Performance par année",
                "",
                "| Année | Trades | Win% | Expectancy R | Total R | PF |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for y, st in deep["by_year"].items():
                if st["n"] == 0:
                    continue
                pfs = ("∞" if not np.isfinite(st["profit_factor"])
                       else f"{st['profit_factor']:.2f}")
                lines.append(
                    f"| {y} | {st['n']} | {st['win_rate']*100:.1f}% | "
                    f"{st['expectancy_r']:+.3f} | {st['total_r']:+.2f} | {pfs} |"
                )
            lines.append("")

            # Per-tier
            lines += [
                "### 3.4 Performance par tier",
                "",
                "| Tier | Trades | Win% | Expectancy R | Total R | PF |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            order = ["PREMIUM", "STANDARD", "WEAK", "INVALID"]
            for tier in order:
                if tier not in deep["by_tier"]:
                    continue
                st = deep["by_tier"][tier]
                if st["n"] == 0:
                    continue
                pfs = ("∞" if not np.isfinite(st["profit_factor"])
                       else f"{st['profit_factor']:.2f}")
                lines.append(
                    f"| {tier} | {st['n']} | {st['win_rate']*100:.1f}% | "
                    f"{st['expectancy_r']:+.3f} | {st['total_r']:+.2f} | {pfs} |"
                )
            lines.append("")

            # Per-hour
            lines += [
                "### 3.5 Performance par heure d'entrée (UTC)",
                "",
                "| Heure | Trades | Win% | Expectancy R | Total R |",
                "|---:|---:|---:|---:|---:|",
            ]
            for h, st in deep["by_hour"].items():
                if st["n"] < 5:
                    continue
                lines.append(
                    f"| {h:02d}h | {st['n']} | {st['win_rate']*100:.1f}% | "
                    f"{st['expectancy_r']:+.3f} | {st['total_r']:+.2f} |"
                )
            lines.append("")

        # Exit reasons
        lines += [
            "### 3.6 Raisons de sortie",
            "",
            "| Raison | Nombre | % |",
            "|---|---:|---:|",
        ]
        total_exits = sum(res.exits_by_reason.values()) or 1
        for reason, n in sorted(
            res.exits_by_reason.items(), key=lambda kv: -kv[1],
        ):
            lines.append(
                f"| {reason} | {n} | {100.0 * n / total_exits:.1f}% |"
            )
        lines.append("")

        lines += [
            "### 3.7 Cadence et machine à états",
            "",
            f"- Signaux/jour : {res.signals_per_day:.2f}",
            f"- Bars moyennes en position : {m.avg_bars_held:.1f} (~"
            f"{m.avg_bars_held * 15:.0f} min)",
            f"- Taux de confirmation : "
            f"{(res.confirmation_rate or 0) * 100:.1f}% "
            f"({res.arms_confirmed}/{res.arms_started} arms confirmés, "
            f"{res.arms_aborted} abandonnés)",
            f"- Signaux générés par le détecteur : "
            f"{res.signals_produced_by_detector:,}",
            f"- Score max observé : {res.score_max:.1f}",
            "",
        ]

    # ----- Problems found -----
    lines += [
        "---",
        "",
        "## 4. Problèmes identifiés & priorisation",
        "",
    ]

    # Problem 1
    lines += [
        "### P1 — [BLOQUANT] Seuil de production inatteignable",
        "",
        "**Constat** : la config par défaut `enter=75, exit=55` produit **0 trades "
        "sur 7 ans** parce que News (20pts) et Volume (10pts) sont toujours nuls en "
        "replay, plafonnant le score à ~70 max. En live, sans un flux news continu, "
        "le même plafond existe.",
        "",
        "**Impact commercial** : un client abonné aujourd'hui ne reçoit aucun signal. ",
        "",
        "**Solutions** :",
        "1. Baisser `enter` à 50-55 pour une config production réaliste.",
        "2. Re-normaliser les poids (retirer News/Volume des composants quand les "
        "données sont absentes, re-répartir sur les autres composants pour que le "
        "score conserve sa plage 0-100).",
        "3. Rendre News/Volume **optionnels** : si absents, score recalé sur les "
        "5 composants restants (BOS 15, FVG 15, OB 10, Régime 25, Momentum+RSI 5 "
        "= 70pts). Diviser par 70, multiplier par 100 pour normaliser.",
        "",
    ]

    # Problem 2: BOS 100% of bars
    lines += [
        "### P2 — BOS_SIGNAL = trend-state, pas event-only",
        "",
        "**Constat** : le détecteur utilise `BOS_SIGNAL` (le state propagé après "
        "une cassure, vrai 100% du temps après la 1ère cassure) comme gate de "
        "direction. Chaque bar peut donc potentiellement produire un signal, "
        "au lieu de seulement les bars où une cassure se produit. C'est "
        "documenté dans le code comme intentionnel (continuation signals) mais "
        "cela **dilue fortement l'edge** : les signaux \"frais\" (BOS_EVENT) "
        "scorent 85% du poids BOS, les continuations 50%.",
        "",
        "**Solutions** :",
        "1. Ajouter un composant \"pullback\" : entrer seulement quand le prix "
        "retouche la structure (OB ou FVG) après la cassure.",
        "2. Augmenter la différence quality (continuation 0.3 au lieu de 0.5) "
        "pour pénaliser plus fort les signaux non-frais.",
        "3. Ne trader que les `BOS_EVENT` frais + confirmation OB/FVG.",
        "",
    ]

    # Problem 3: News replay
    lines += [
        "### P3 — News bypassée en backtest",
        "",
        "**Constat** : le replay ne charge pas le calendrier économique "
        "(`economic_calendar_HIGH_IMPACT_2019_2025.csv` existe pourtant dans "
        "`/data/`). Tous les signaux pendant NFP / FOMC / CPI sont évalués "
        "comme s'il n'y avait pas d'événement.",
        "",
        "**Solutions** :",
        "1. Charger le CSV economic_calendar et construire un `NewsAssessment` "
        "synthétique par bar (blocking window ±30 min autour high-impact).",
        "2. Comparer le edge \"news-aware\" vs \"news-blind\" — c'est un gros "
        "argument marketing.",
        "",
    ]

    # Problem 4: No slippage/commission
    lines += [
        "### P4 — Coûts de transaction non modélisés",
        "",
        "**Constat** : les PnL sont calculés `exit_price - entry_price` sans "
        "spread, commission ni slippage. Sur XAU/USD M15 avec ~50 trades/an, "
        "spread moyen 20-30 pips × 0.01 = ~$0.25/trade, slippage similaire. "
        "Coût réel ~$0.50-1.00 par round-trip.",
        "",
        "**Solutions** :",
        "1. Ajouter `commission_per_trade` et `spread_pips` en paramètres du "
        "harness. Soustraire du PnL.",
        "2. Refaire le sweep avec frais réalistes — certaines configs "
        "positives deviendront négatives.",
        "",
    ]

    # Problem 5: No regime breakdown
    lines += [
        "### P5 — Pas de segmentation par régime",
        "",
        "**Constat** : le rapport actuel ne ventile pas la performance par régime "
        "(trending vs ranging, high-vol vs low-vol). Or la stratégie peut être "
        "profitable en trend et perdante en range, ou inversement.",
        "",
        "**Solutions** :",
        "1. Tagger chaque trade avec le régime+vol_regime à l'entrée, ventiler "
        "les métriques par (trend×vol).",
        "2. Désactiver automatiquement les trades dans les régimes où l'edge "
        "est négatif.",
        "",
    ]

    # Problem 6: Fixed RR
    lines += [
        "### P6 — RR fixe 2:1 peu adaptatif",
        "",
        "**Constat** : SL = 2×ATR, TP = 4×ATR (ratio 2:1) quelle que soit la "
        "situation. En fort trend, TP trop proche (rate les moves de 5-8 ATR). "
        "En range, TP trop loin (rarement atteint, sorties par time-expiry).",
        "",
        "**Solutions** :",
        "1. RR adaptatif : 1.5:1 en ranging, 3:1 en strong trend.",
        "2. Trailing stop après +1R : lock in profit sans plafonner le run.",
        "",
    ]

    # Summary verdict
    lines += [
        "---",
        "",
        "## 5. Verdict commercialisation",
        "",
    ]

    if best_payload is None:
        lines += [
            "### ❌ **NON commercialisable en l'état**",
            "",
            "La stratégie ne produit aucun trade significatif sur 7 ans de "
            "données historiques avec aucune config testée. Le problème n'est "
            "même pas un edge négatif — c'est un système inerte. **Corriger P1 "
            "est absolument prioritaire** avant toute autre chose.",
            "",
        ]
    else:
        m = best_payload["metrics"]
        pf = m.profit_factor
        exp = m.expectancy_r
        dd = m.max_drawdown_r
        if pf >= 1.3 and exp > 0.05:
            verdict = "✅ **Prometteur avec ajustements**"
            detail = (
                "L'edge sous-jacent existe. Avec les corrections P1+P3+P4 "
                "(ré-normalisation du score, intégration news, modélisation "
                "des frais), le produit peut être commercialisé avec un "
                "message réaliste sur les attentes."
            )
        elif pf >= 1.0:
            verdict = "⚠️ **Marginal — ne pas commercialiser tel quel**"
            detail = (
                "La stratégie est légèrement profitable hors frais, mais le "
                "profit factor trop bas ne laisse aucune marge pour spread + "
                "slippage. Il faut resserrer l'entry (pullback filter P2) "
                "avant de pouvoir prendre un client payant."
            )
        else:
            verdict = "❌ **NON commercialisable**"
            detail = (
                f"Profit factor {pf:.2f} et expectancy {exp:+.3f} R signifient "
                "que le système perd de l'argent en moyenne. Corriger P1 (seuils) "
                "ne suffira pas — il faut revoir la logique de signal "
                "(pullback filter P2) avant de re-backtester."
            )
        lines += [
            f"### {verdict}",
            "",
            detail,
            "",
            "**Plan d'action recommandé avant déploiement :**",
            "1. **[P1]** Re-normaliser le scoring pour tenir compte des composants "
            "absents — sinon aucun client ne recevra jamais rien.",
            "2. **[P4]** Ajouter la modélisation des coûts et re-tester.",
            "3. **[P2]** Ajouter un filtre pullback : n'entrer que sur retest de "
            "structure après cassure.",
            "4. **[P3]** Intégrer le calendrier économique (déjà en `/data/`).",
            "5. **[P5]** Ventiler par régime, désactiver les trades dans les "
            "régimes négatifs.",
            "6. Re-lancer cet audit après chaque correction pour mesurer le gain.",
            "",
        ]

    lines += [
        "---",
        "",
        f"_Rapport généré automatiquement. Fichiers associés :_",
        "- `reports/audit/sweep_results.json` — données brutes du sweep",
        "- `reports/audit/trades_combined.csv` — trades de la meilleure config",
        "- `reports/audit/score_distribution.csv` — échantillons de scores",
        "",
    ]

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        logger.error("CSV not found: %s", CSV_PATH)
        return 2

    enriched = load_and_enrich(CSV_PATH)
    date_range = (str(enriched.index[WARMUP]), str(enriched.index[-1]))
    bars_total = len(enriched)

    # News provider: loads HIGH-impact USD/XAU events so the score's News
    # component is no longer structurally 0 in backtest. Absent file or
    # missing CSV still works — BacktestNewsProvider.at() returns None and
    # the detector's P1 renormalisation kicks in.
    news_provider: Optional[BacktestNewsProvider] = None
    if CALENDAR_CSV.exists():
        news_provider = BacktestNewsProvider.from_csv(
            CALENDAR_CSV, symbol=SYMBOL,
        )
        logger.info(
            "Loaded %d HIGH-impact events from %s (±30 min blackout)",
            len(news_provider.events), CALENDAR_CSV,
        )
    else:
        logger.warning(
            "Calendar CSV %s not found — running audit without news blackout",
            CALENDAR_CSV,
        )

    # 1. Full score distribution (one pass) — or reuse if already captured.
    # Cache is keyed on (presence of news provider): with news, blackout
    # bars are suppressed so the distribution will differ from a news-blind
    # run and we must not reuse a stale cache.
    score_csv = OUT_DIR / (
        "score_distribution_news.csv" if news_provider is not None
        else "score_distribution.csv"
    )
    if score_csv.exists():
        logger.info("Reusing cached score distribution from %s", score_csv)
        _df = pd.read_csv(score_csv)
        scores = [
            ScoreSample(ts=str(row.ts), score=float(row.score),
                        direction=str(row.direction))
            for row in _df.itertuples(index=False)
        ]
        logger.info("Loaded %d cached scored bars", len(scores))
    else:
        scores = capture_full_score_distribution(
            enriched, news_provider=news_provider,
        )
        if scores:
            pd.DataFrame([
                {"ts": s.ts, "score": s.score, "direction": s.direction}
                for s in scores
            ]).to_csv(score_csv, index=False)

    # 2. Sweep
    sweep = run_sweep(enriched, news_provider=news_provider)

    # 3. Save raw sweep
    raw: Dict[str, dict] = {}
    for label, p in sweep.items():
        res: ReplayResults = p["results"]
        m = p["metrics"]
        raw[label] = {
            "enter": p["enter"],
            "exit": p["exit"],
            "summary": {
                "total_trades": m.total_trades,
                "win_rate": m.win_rate,
                "expectancy_r": m.expectancy_r,
                "total_r": m.total_r,
                "profit_factor": (m.profit_factor
                                  if np.isfinite(m.profit_factor)
                                  else "inf"),
                "max_drawdown_r": m.max_drawdown_r,
                "sharpe_annualised": m.sharpe_annualised,
                "sortino_annualised": m.sortino_annualised,
                "trades_per_year": m.trades_per_year,
            },
            "exits_by_reason": dict(res.exits_by_reason),
            "score_max": res.score_max,
        }
    (OUT_DIR / "sweep_results.json").write_text(
        json.dumps(raw, indent=2, default=str),
        encoding="utf-8",
    )

    # 4. Export trades for best config
    best_label = None
    best_pf = -1.0
    for label, p in sweep.items():
        m = p["metrics"]
        if (m.total_trades >= 30
            and np.isfinite(m.profit_factor)
            and m.profit_factor > best_pf):
            best_pf = m.profit_factor
            best_label = label
    if best_label is None:
        # Pick the config with the most trades
        candidates = [(l, p) for l, p in sweep.items()
                      if p["metrics"].total_trades > 0]
        if candidates:
            best_label = max(
                candidates, key=lambda kv: kv[1]["metrics"].total_trades,
            )[0]

    if best_label is not None:
        best_trades = sweep[best_label]["results"].trades
        if best_trades:
            pd.DataFrame([t.to_dict() for t in best_trades]).to_csv(
                OUT_DIR / "trades_combined.csv", index=False,
            )
            logger.info(
                "Wrote %d trades from config '%s' to trades_combined.csv",
                len(best_trades), best_label,
            )

    # 5. Render report
    md = render_markdown(sweep, scores, date_range, bars_total)
    (OUT_DIR / "audit_report.md").write_text(md, encoding="utf-8")
    logger.info("Report written to %s", OUT_DIR / "audit_report.md")

    # Echo verdict to stdout
    print()
    print("=" * 76)
    print(" AUDIT COMPLETE")
    print("=" * 76)
    print(f" Report : {OUT_DIR / 'audit_report.md'}")
    print(f" Sweep  : {OUT_DIR / 'sweep_results.json'}")
    print(f" Trades : {OUT_DIR / 'trades_combined.csv'}")
    print(f" Scores : {OUT_DIR / 'score_distribution.csv'}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
