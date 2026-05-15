"""Walk-forward backtest skeleton — Smart Sentinel AI (Eval 18, K3).

PURPOSE
-------
Provide an honest, prospect-defensible PnL evaluation that eliminates
in-sample over-fitting and threshold-shopping. Splits XAU M15 history
2019-01 -> 2025-12 into THREE non-overlapping folds with explicit purge
+ embargo gaps, tunes the state-machine thresholds on TRAIN+VAL only,
and reports OOS performance on TEST as the only number that may appear
on landing pages.

Why this matters
----------------
The current ``scripts/audit_backtest.py`` runs a 7-config sweep over the
full 7-year window and headlines the BEST profit factor observed. That
is multiple-testing biased: with 7 configs the expected best-of-7 PF
under H0 (no edge) is mechanically inflated. White's Reality Check or
Hansen SPA must be applied to that sweep, OR the sweep must be confined
to TRAIN+VAL with the chosen config frozen and replayed on TEST.

This skeleton implements the second option (simpler, defensible).

USAGE
-----
::

    python reports/eval_18_walkforward_skeleton.py \\
        --csv data/XAU_15MIN_2019_2026.csv \\
        --calendar data/economic_calendar_HIGH_IMPACT_2019_2025.csv \\
        --out reports/walkforward_xau_m15.json

CRITICAL — read before deleting anything
----------------------------------------
*Until this script has been executed end-to-end and the result reviewed,
no PF / Sharpe / win-rate number derived from* ``audit_backtest.py``
*may appear on the public landing page. See* ``BACKTEST_LEGAL_GUARDRAILS.md``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Make sure src/ is importable when invoked as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.metrics import compute_performance, tier_from_score
from src.backtest.news_replay import BacktestNewsProvider
from src.backtest.state_machine_replay import SignalReplay
from src.environment.strategy_features import SmartMoneyEngine
from src.intelligence.confluence_detector import ConfluenceDetector
from src.intelligence.signal_state_machine import StateMachineConfig

logger = logging.getLogger("walkforward")


# =============================================================================
# FOLD DEFINITION
# =============================================================================
# Anchored, expanding-train walk-forward with 5-day embargo (= 480 M15 bars).
# Embargo prevents intra-day signal leakage when SmartMoneyEngine carries
# state across the boundary (FVG / OB lookbacks reach back ~50-200 bars).
#
# Anti-leakage budget:
#   * ZSCORE rolling window in environment.py = 200 bars  -> 50 hours
#   * SMC fractal window 2*N+1 with center=True  -> +2 bars
#   * BOS retest lookback  -> ~50 bars max
# 5-day (480 bars) embargo absorbs all of these comfortably.

EMBARGO_BARS = 480  # 5 trading days at M15

@dataclass(frozen=True)
class Fold:
    name: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

DEFAULT_FOLD = Fold(
    name="xau_m15_2019_2025",
    train_start="2019-01-03",
    train_end="2022-12-31",
    val_start="2023-01-08",     # +5 trading days embargo from train_end
    val_end="2023-12-31",
    test_start="2024-01-08",    # +5 trading days embargo from val_end
    test_end="2025-12-31",
)


# =============================================================================
# PARAMETER SEARCH SPACE
# =============================================================================
# Restricted to parameters with a clear economic interpretation. We do NOT
# sweep cooldown/confirmation: those are state-machine ergonomics, not edge.

# enter / exit / max_age — the only knobs prospects actually care about.
SEARCH_GRID: Dict[str, List[float]] = {
    "enter":   [40, 45, 50, 55, 60],
    "exit":    [25, 30, 35, 40],
    "max_age": [8, 12, 16, 24],
}

# We require enter > exit (hysteresis) and trim degenerate combos.
def _enumerate_params() -> Iterable[Dict[str, float]]:
    for e, x, m in itertools.product(*SEARCH_GRID.values()):
        if e <= x:
            continue
        yield {"enter": float(e), "exit": float(x), "max_age": int(m)}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_enrich(csv_path: Path) -> pd.DataFrame:
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
    logger.info("Loaded %d bars  %s -> %s", len(df), df.index[0], df.index[-1])

    # IMPORTANT: SmartMoneyEngine.detect_fractals uses rolling(center=True)
    # which is a look-ahead operator before the explicit shift(N) is applied
    # at strategy_features.py:637-638.  The shift restores causality but ONLY
    # within the global frame. If we enrich AFTER slicing the fold, the
    # boundary bars at the start of each fold get NaN fractals (acceptable
    # behaviour, no leakage).  If we enrich BEFORE slicing, fractals near
    # the train/val boundary used `high.iloc[i+N]` from the validation
    # window — that IS leakage.  We therefore enrich each fold separately.
    return df


def slice_fold(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
    sub = df.loc[mask].copy()
    if len(sub) < 1000:
        raise ValueError(f"Fold {start}->{end} has only {len(sub)} bars")
    return sub


def enrich_fold(df_slice: pd.DataFrame) -> pd.DataFrame:
    engine = SmartMoneyEngine(data=df_slice, config={}, verbose=False)
    enriched = engine.analyze()
    # First EMBARGO_BARS bars of the slice may have unstable rolling
    # features — drop them to avoid feeding low-quality bars to the
    # state machine during early evaluation.
    if len(enriched) <= EMBARGO_BARS:
        raise ValueError(f"Slice too short for embargo: {len(enriched)} <= {EMBARGO_BARS}")
    return enriched.iloc[EMBARGO_BARS:].copy()


# =============================================================================
# SINGLE BACKTEST RUN
# =============================================================================

def run_one(
    enriched: pd.DataFrame,
    params: Dict[str, float],
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
    news_provider: Optional[BacktestNewsProvider] = None,
) -> Dict[str, float]:
    """Run replay with a single (enter, exit, max_age) combo, return metrics."""
    cfg = StateMachineConfig(
        symbol=symbol,
        enter_threshold=float(params["enter"]),
        exit_threshold=float(params["exit"]),
        confirm_bars=2,
        cooldown_bars=2,
        max_signal_age_bars=int(params["max_age"]),
    )
    detector = ConfluenceDetector(
        symbol=symbol,
        min_score=min(cfg.enter_threshold, cfg.exit_threshold),
        require_retest=True,
    )
    replay = SignalReplay(
        symbol=symbol,
        timeframe=timeframe,
        state_machine_config=cfg,
        confluence_detector=detector,
        warmup_bars=100,
        news_provider=news_provider,
    )
    res = replay.run(enriched)
    metrics = compute_performance(
        res.trades, timeframe=timeframe,
        tier_fn=lambda t: tier_from_score(t.confluence_score),
        bars_processed=res.bars_processed,
    )
    pf = metrics.profit_factor if math.isfinite(metrics.profit_factor) else 1e9
    return {
        "params": params,
        "trades": metrics.total_trades,
        "win_rate": metrics.win_rate,
        "expectancy_r": metrics.expectancy_r,
        "total_r": metrics.total_r,
        "profit_factor": pf,
        "max_drawdown_r": metrics.max_drawdown_r,
        "sharpe_annualised": metrics.sharpe_annualised,
        "sortino_annualised": metrics.sortino_annualised,
        "trades_per_year": metrics.trades_per_year,
        "calmar": metrics.calmar,
    }


# =============================================================================
# OBJECTIVE FOR TUNING
# =============================================================================

def score_run(metrics: Dict[str, float], min_trades: int = 50) -> float:
    """Multi-objective scalar used to pick the winning config on TRAIN+VAL.

    Penalises configs with too few trades (under-powered evidence).
    Combines PF and annualised Sharpe; both must beat 1.0 / 0.5 to be
    considered viable — otherwise returns -inf.
    """
    if metrics["trades"] < min_trades:
        return -math.inf
    if metrics["profit_factor"] <= 1.0:
        return -math.inf
    sharpe = metrics["sharpe_annualised"] or 0.0
    if sharpe <= 0.5:
        return -math.inf
    return float(np.log(metrics["profit_factor"]) + 0.5 * sharpe
                 - 0.1 * metrics["max_drawdown_r"] / max(1e-9, metrics["total_r"]))


# =============================================================================
# WALK-FORWARD ORCHESTRATION
# =============================================================================

def walk_forward(
    full_df: pd.DataFrame,
    fold: Fold,
    news_provider: Optional[BacktestNewsProvider] = None,
) -> Dict[str, object]:
    """Train/val/test pipeline with explicit purge + embargo."""
    logger.info("Fold %s", fold.name)

    # 1. Slice raw, enrich each independently (avoid cross-fold lookahead).
    train_raw = slice_fold(full_df, fold.train_start, fold.train_end)
    val_raw   = slice_fold(full_df, fold.val_start,   fold.val_end)
    test_raw  = slice_fold(full_df, fold.test_start,  fold.test_end)

    train = enrich_fold(train_raw)
    val   = enrich_fold(val_raw)
    test  = enrich_fold(test_raw)

    logger.info("Bars after embargo: train=%d val=%d test=%d",
                len(train), len(val), len(test))

    # 2. Tune thresholds on TRAIN, validate on VAL, pick best by combined score.
    candidates: List[Dict[str, object]] = []
    for params in _enumerate_params():
        train_m = run_one(train, params, news_provider=news_provider)
        val_m   = run_one(val,   params, news_provider=news_provider)
        # Selection uses VAL only — train metrics are advisory.
        val_score = score_run(val_m)
        candidates.append({
            "params": params,
            "train": train_m,
            "val": val_m,
            "val_score": val_score,
        })
        logger.info("  params=%s  trainPF=%.2f  valPF=%.2f  valScore=%.3f",
                    params, train_m["profit_factor"],
                    val_m["profit_factor"], val_score)

    candidates.sort(key=lambda c: c["val_score"], reverse=True)
    if not candidates or candidates[0]["val_score"] == -math.inf:
        logger.warning("No config met VAL gate (PF>1.0, Sharpe>0.5).")
        winner = None
    else:
        winner = candidates[0]
        logger.info("WINNER on VAL: %s  valPF=%.2f  valSharpe=%.2f",
                    winner["params"], winner["val"]["profit_factor"],
                    winner["val"]["sharpe_annualised"])

    # 3. Replay TEST with winner's params — this is the OOS number.
    test_m: Optional[Dict[str, float]] = None
    if winner is not None:
        test_m = run_one(test, winner["params"], news_provider=news_provider)
        logger.info("OOS TEST: %s", test_m)

    return {
        "fold": asdict(fold),
        "candidates": candidates,
        "winner": winner,
        "test": test_m,
        "n_candidates": len(candidates),
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/XAU_15MIN_2019_2026.csv")
    p.add_argument("--calendar", default="data/economic_calendar_HIGH_IMPACT_2019_2025.csv")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--out", default="reports/walkforward_xau_m15.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        return 2

    df = load_and_enrich(csv_path)

    cal_path = Path(args.calendar)
    news_provider = None
    if cal_path.exists():
        news_provider = BacktestNewsProvider.from_csv(cal_path, symbol=args.symbol)
        logger.info("News provider loaded: %d events", len(news_provider.events))
    else:
        logger.warning("Calendar CSV %s missing — running news-blind", cal_path)

    result = walk_forward(df, DEFAULT_FOLD, news_provider=news_provider)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info("Walk-forward report -> %s", out_path)

    print()
    print("=" * 76)
    print(" WALK-FORWARD COMPLETE")
    print("=" * 76)
    if result["winner"] is None:
        print(" Result: no config passed VAL gate (PF>1.0 + Sharpe>0.5).")
        print(" -> NOT commercialisable. Do NOT publish any PF on landing.")
    else:
        w = result["winner"]
        t = result["test"]
        print(f" Winner params : {w['params']}")
        print(f" VAL  PF/Sharpe: {w['val']['profit_factor']:.2f} / "
              f"{w['val']['sharpe_annualised']:.2f}")
        print(f" TEST PF/Sharpe: {t['profit_factor']:.2f} / "
              f"{t['sharpe_annualised']:.2f}  ({t['trades']} trades)")
        if t["profit_factor"] >= 1.5 and t["sharpe_annualised"] >= 0.8:
            print(" -> Commercialisable with explicit OOS framing (see guardrails).")
        elif t["profit_factor"] >= 1.2:
            print(" -> Borderline: market as 'paper-trade beta' only.")
        else:
            print(" -> Not commercialisable yet. Iterate on signal logic.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
