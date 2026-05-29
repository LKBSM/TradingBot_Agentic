"""Phase 2 — Empirical validation of HTF alignment as a PnL filter.

Runs the canonical SignalReplay on XAU M15 2019-2025, computes the
``MultiTimeframeFeatures`` alignment label at each entry bar, bins trades
by quality band, and bootstraps CI95 on the profit-factor distribution
for the "filtered" (drop H4 counter-trend) and "all" strategies.

**Decision gate (Phase 1 plan)**
  If PF_lo (CI95) of the "filtered" strategy rises by ≥0.05 vs the "all"
  strategy → recommend GO for Phase 3 (lift htf_alignment weight 0 → 13).
  Otherwise → keep the readout descriptive only, weight stays at 0.

Usage
-----
    python -m scripts.eval_mtf_alignment
    python -m scripts.eval_mtf_alignment --csv data/XAU_15MIN_2019_2025.csv \
        --bootstrap-iters 5000 --out reports/eval_mtf_alignment.md

Outputs
-------
  reports/eval_mtf_alignment.csv  — per-trade alignment label + PnL
  reports/eval_mtf_alignment.md   — bucketed PF table + bootstrap CI95 + verdict
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.backtest.state_machine_replay import SignalReplay, TradeRecord
from src.environment.multi_timeframe_features import MultiTimeframeFeatures
from src.environment.strategy_features import SmartMoneyEngine
from src.intelligence.signal_state_machine import StateMachineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_mtf_alignment")


# --------------------------------------------------------------------------- #
# Data loading (mirrors eval_07_state_machine_sweep helper)
# --------------------------------------------------------------------------- #

def load_and_enrich(csv_path: Path) -> pd.DataFrame:
    log.info("Loading %s", csv_path)
    df = pd.read_csv(csv_path)
    for cand in ("timestamp", "Date", "date", "datetime", "time"):
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand])
            df = df.set_index(cand)
            break
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("open", "high", "low", "close", "volume"):
            rename[col] = lc.capitalize()
    if rename:
        df = df.rename(columns=rename)
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    log.info("Loaded %d bars (%s -> %s)", len(df), df.index.min(), df.index.max())

    log.info("Running SMC enrichment …")
    engine = SmartMoneyEngine(
        data=df,
        config={
            "RSI_WINDOW": 14, "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
            "BB_WINDOW": 20, "ATR_WINDOW": 14,
            "FRACTAL_WINDOW": 2, "FVG_THRESHOLD": 0.1,
            "OB_REQUIRE_FVG": False,
        },
    )
    enriched = engine.analyze()
    log.info("Enriched %d bars (after NaN drop)", len(enriched))
    return enriched


# --------------------------------------------------------------------------- #
# HTF alignment label — mirrors ConfluenceDetector._score_htf_alignment
# --------------------------------------------------------------------------- #

def htf_alignment_label(
    htf_features: Optional[Dict[str, float]],
    direction: str,
) -> Tuple[str, float]:
    """Return (label, quality) for a given HTF features dict and direction.

    Labels:
      'counter_h4'      → H4 trend against direction (would drop signal)
      'full_alignment'  → H1 + H4 both aligned
      'h4_aligned'      → H4 aligned, H1 neutral or opposite
      'h1_aligned'      → H1 aligned, H4 neutral
      'ranging'         → both H1 and H4 neutral
      'misaligned'      → other mismatch
      'na'              → no HTF features available
    """
    if htf_features is None:
        return "na", 0.0
    trend_1h = float(htf_features.get("HTF_TREND_1H", 0.0))
    trend_4h = float(htf_features.get("HTF_TREND_4H", 0.0))
    strength_1h = float(htf_features.get("HTF_STRENGTH_1H", 0.0))
    strength_4h = float(htf_features.get("HTF_STRENGTH_4H", 0.0))

    required = 1.0 if direction.upper() == "LONG" else -1.0
    h4_aligned = trend_4h == required
    h4_counter = trend_4h == -required
    h1_aligned = trend_1h == required
    both_neutral = trend_4h == 0.0 and trend_1h == 0.0

    if h4_counter:
        return "counter_h4", 0.0
    if h4_aligned and h1_aligned:
        return "full_alignment", 0.7 + 0.3 * min(1.0, max(strength_4h, strength_1h))
    if h4_aligned:
        return "h4_aligned", 0.5 + 0.2 * min(1.0, strength_4h)
    if h1_aligned:
        return "h1_aligned", 0.4
    if both_neutral:
        return "ranging", 0.3
    return "misaligned", 0.0


# --------------------------------------------------------------------------- #
# Bootstrap CI95 on PF
# --------------------------------------------------------------------------- #

def _profit_factor(r_series: np.ndarray) -> float:
    wins = r_series[r_series > 0].sum()
    losses = -r_series[r_series < 0].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def bootstrap_pf_ci95(
    r_series: List[float],
    iters: int = 5000,
    seed: int = 2026,
) -> Tuple[float, float, float, float]:
    """Bootstrap (point PF, mean PF, lo CI95, hi CI95) on the R series."""
    if not r_series:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(r_series, dtype=float)
    point = _profit_factor(arr)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = np.empty(iters)
    for i in range(iters):
        sample = arr[rng.integers(0, n, n)]
        boots[i] = _profit_factor(sample)
    finite = boots[np.isfinite(boots)]
    if len(finite) == 0:
        return point, 0.0, 0.0, 0.0
    lo = float(np.percentile(finite, 2.5))
    hi = float(np.percentile(finite, 97.5))
    return point, float(finite.mean()), lo, hi


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def label_trades(
    trades: List[TradeRecord],
    mtf: MultiTimeframeFeatures,
    df_index: pd.DatetimeIndex,
) -> List[Dict[str, object]]:
    """For each trade, look up HTF features at entry and assign a label."""
    out: List[Dict[str, object]] = []
    index_pos = {ts: i for i, ts in enumerate(df_index)}
    for t in trades:
        try:
            ts = pd.Timestamp(t.entry_bar)
            idx = index_pos.get(ts)
            if idx is None:
                # Try without tz / with tz round-trip
                idx = index_pos.get(ts.tz_localize(None) if ts.tzinfo else ts)
            htf = mtf.get_features(idx) if idx is not None else None
        except Exception:
            htf = None
        label, quality = htf_alignment_label(htf, t.direction)
        out.append({
            "entry_bar": t.entry_bar,
            "direction": t.direction,
            "exit_reason": t.exit_reason,
            "confluence_score": t.confluence_score,
            "r_multiple": t.r_multiple,
            "alignment_label": label,
            "alignment_quality": round(quality, 3),
            "h4_trend": (htf or {}).get("HTF_TREND_4H", 0.0),
            "h1_trend": (htf or {}).get("HTF_TREND_1H", 0.0),
            "h4_strength": (htf or {}).get("HTF_STRENGTH_4H", 0.0),
            "h1_strength": (htf or {}).get("HTF_STRENGTH_1H", 0.0),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=_PROJECT_ROOT / "data" / "XAU_15MIN_2019_2025.csv",
    )
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--enter-threshold", type=float, default=55.0)
    parser.add_argument("--exit-threshold", type=float, default=35.0)
    parser.add_argument(
        "--disable-vol-regime",
        action="store_true",
        default=True,
        help=(
            "Disable high-vol forced exit (default) — empirically the vol "
            "classifier marks the bar after entry as 'high' and triggers an "
            "instant REGIME_SHIFTED exit, collapsing the sample. "
            "Phase 2 evaluates HTF *as a filter*, not the vol regime."
        ),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=_PROJECT_ROOT / "reports" / "eval_mtf_alignment.csv",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=_PROJECT_ROOT / "reports" / "eval_mtf_alignment.md",
    )
    args = parser.parse_args()

    enriched = load_and_enrich(args.csv)

    # Fit MTF features on the same OHLCV (capitalised columns expected by
    # the resampler — strategy_features returns lowercase, so build a view).
    log.info("Fitting MultiTimeframeFeatures (resamples 1h + 4h, ~5s) …")
    mtf_df = enriched[["open", "high", "low", "close", "volume"]].rename(
        columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        }
    )
    mtf = MultiTimeframeFeatures(base_timeframe="15min").fit(mtf_df)

    log.info(
        "Running SignalReplay (enter=%.0f / exit=%.0f) …",
        args.enter_threshold, args.exit_threshold,
    )
    cfg = StateMachineConfig(
        symbol="XAUUSD",
        enter_threshold=float(args.enter_threshold),
        exit_threshold=float(args.exit_threshold),
        high_vol_forces_exit=not args.disable_vol_regime,
    )
    replay = SignalReplay(
        symbol="XAUUSD", timeframe="M15",
        state_machine_config=cfg,
        use_regime=True,
        use_vol_regime=not args.disable_vol_regime,
    )
    results = replay.run(enriched)
    log.info(
        "Replay done: %d trades, PF=%.3f, signals_per_day=%.2f",
        results.total_trades, results.profit_factor, results.signals_per_day,
    )

    if not results.trades:
        log.warning(
            "Zero trades produced — cannot evaluate HTF filter. "
            "Decision gate: NO-GO (insufficient sample)."
        )
        return 2

    labelled = label_trades(results.trades, mtf, enriched.index)

    # Write per-trade CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(labelled[0].keys()))
        writer.writeheader()
        writer.writerows(labelled)
    log.info("Wrote per-trade CSV: %s", args.out_csv)

    # ----- Bucketed PF per label -----
    by_label: Dict[str, List[float]] = {}
    for row in labelled:
        by_label.setdefault(row["alignment_label"], []).append(row["r_multiple"])

    label_order = [
        "full_alignment", "h4_aligned", "h1_aligned",
        "ranging", "misaligned", "counter_h4", "na",
    ]
    bucket_rows = []
    for lbl in label_order:
        r = by_label.get(lbl, [])
        if not r:
            bucket_rows.append((lbl, 0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        n = len(r)
        wr = sum(1 for x in r if x > 0) / n
        point, mean, lo, hi = bootstrap_pf_ci95(r, iters=args.bootstrap_iters)
        bucket_rows.append((lbl, n, wr, point, mean, lo, hi))

    # ----- Strategy comparison: "all" vs "filtered" (drop counter_h4) -----
    r_all = [row["r_multiple"] for row in labelled]
    r_filtered = [
        row["r_multiple"] for row in labelled
        if row["alignment_label"] != "counter_h4"
    ]

    pf_all = bootstrap_pf_ci95(r_all, iters=args.bootstrap_iters)
    pf_filt = bootstrap_pf_ci95(r_filtered, iters=args.bootstrap_iters)
    pf_lo_rise = pf_filt[2] - pf_all[2]

    decision = "GO Phase 3" if pf_lo_rise >= 0.05 else "NO-GO — keep weight=0"

    # ----- Build markdown report -----
    md = []
    md.append("# Phase 2 — HTF Alignment Empirical Validation\n")
    md.append(f"- CSV  : `{args.csv}`")
    md.append(f"- Bars : {len(enriched):,} ({enriched.index.min()} → {enriched.index.max()})")
    md.append(f"- State machine: enter={args.enter_threshold}, exit={args.exit_threshold}")
    md.append(f"- Total trades (no filter): **{results.total_trades}**")
    md.append(f"- Bootstrap iters: {args.bootstrap_iters}\n")

    md.append("## Bucketed PF by HTF alignment label\n")
    md.append("| Label | n | win_rate | PF (point) | PF (mean) | PF lo CI95 | PF hi CI95 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for lbl, n, wr, point, mean, lo, hi in bucket_rows:
        md.append(
            f"| `{lbl}` | {n} | {wr:.2%} | {point:.3f} | {mean:.3f} | {lo:.3f} | {hi:.3f} |"
        )

    md.append("\n## Strategy comparison: ALL vs FILTERED (drop counter_h4)\n")
    md.append("| Strategy | n | PF (point) | PF (mean) | PF lo CI95 | PF hi CI95 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    md.append(
        f"| ALL       | {len(r_all)} | {pf_all[0]:.3f} | {pf_all[1]:.3f} | "
        f"{pf_all[2]:.3f} | {pf_all[3]:.3f} |"
    )
    md.append(
        f"| FILTERED  | {len(r_filtered)} | {pf_filt[0]:.3f} | {pf_filt[1]:.3f} | "
        f"{pf_filt[2]:.3f} | {pf_filt[3]:.3f} |"
    )

    md.append("\n## Decision gate\n")
    md.append(f"- PF lo CI95 rise: **{pf_lo_rise:+.3f}** (gate: ≥ +0.050)")
    md.append(f"- Sample loss (counter_h4 dropped): {len(r_all) - len(r_filtered)} trades")
    md.append(f"- **Verdict: {decision}**")
    if pf_lo_rise >= 0.05:
        md.append(
            "\n→ Phase 3: lift `htf_alignment` weight 0 → 13, rebalance the other "
            "components to sum to 100, re-sweep PREMIUM/STANDARD/WEAK tier cutpoints."
        )
    else:
        md.append(
            "\n→ Keep `htf_alignment` at weight=0; the readout remains descriptive "
            "only. No behavioral change. Phase 3 is rejected by the empirical gate."
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote MD report: %s", args.out_md)

    # Print ASCII-only summary to stdout (cp1252-safe on Windows consoles).
    summary = "\n".join(md)
    sys.stdout.buffer.write(summary.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    return 0 if pf_lo_rise >= 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
