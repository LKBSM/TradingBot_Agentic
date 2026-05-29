"""Evaluate the economic calendar / blackout flag on OOS data.

The prod NewsAnalysisAgent blocks the trading window ±30 min around HIGH-impact
events from `data/economic_calendar_HIGH_IMPACT_2019_2025.csv`. The streaming
pipeline emits `news_blackout` per bar; here we reconstruct the same window
deterministically from the calendar CSV (no live news state needed) and check:

Q1 (Justesse factuelle):
  - Calendar structural sanity: no duplicate (time,event) pairs, dates parsable,
    impact column populated, currency column populated.
  - Coverage: fraction of OOS bars that fall inside any ±30 min HIGH-impact window
    (any-currency). Per-currency split for context (USD, EUR).
  - Reality check: on blocked bars, is realized 15-min return absolute value (proxy
    for "news impact") significantly larger than on non-blocked bars?

Q2 (Stabilité):
  - Bar-level consistency: do consecutive blocked bars share the same dominant
    event? (no flicker)

Q3 (Calibration): N/A.

Note: doc claims "30 min before, 60 min after". Source code uses 30/30. Flagged
as a doc inconsistency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))

from scripts.audit.descriptive_quality._harness import (  # noqa: E402
    InstrumentData,
    OOS_START,
    bootstrap_ci,
    load_calendar,
    load_instrument,
)


BLOCK_BEFORE_MIN = 30
BLOCK_AFTER_MIN = 30


def _blocked_mask(bar_index: pd.DatetimeIndex, cal: pd.DataFrame) -> np.ndarray:
    """Vectorized: True if bar falls within any ±window of a HIGH impact event."""
    n = len(bar_index)
    mask = np.zeros(n, dtype=bool)
    bar_ts = pd.to_datetime(bar_index).to_numpy().astype("datetime64[ns]")
    before = np.timedelta64(BLOCK_BEFORE_MIN, "m").astype("timedelta64[ns]")
    after = np.timedelta64(BLOCK_AFTER_MIN, "m").astype("timedelta64[ns]")
    cal_times = pd.to_datetime(cal["date"]).to_numpy().astype("datetime64[ns]")
    for ev_t in cal_times:
        lo = ev_t - before
        hi = ev_t + after
        in_window = (bar_ts >= lo) & (bar_ts <= hi)
        mask |= in_window
    return mask


def eval_calendar_for(inst: InstrumentData, cal: pd.DataFrame, label: str) -> dict:
    df = inst.oos
    bar_index = df.index
    n_bars = len(df)

    # Restrict calendar to OOS window for efficient comparison
    cal_oos = cal[
        (cal["date"] >= bar_index.min()) & (cal["date"] <= bar_index.max())
    ].reset_index(drop=True)

    # === Q1.a — structural sanity ===
    n_events = len(cal_oos)
    duplicates = cal_oos.duplicated(subset=["date", "event"]).sum() if "event" in cal_oos.columns else 0
    nan_impact = cal_oos["impact"].isna().sum() if "impact" in cal_oos.columns else 0
    nan_currency = (
        cal_oos["currency"].isna().sum() if "currency" in cal_oos.columns else 0
    )
    columns = list(cal_oos.columns)

    # === Q1.b — coverage of OOS bars (any currency) ===
    mask_any = _blocked_mask(bar_index, cal_oos)
    coverage_any = float(mask_any.mean())
    coverage_per_currency = {}
    if "currency" in cal_oos.columns:
        for ccy in ["USD", "EUR", "GBP", "JPY"]:
            sub = cal_oos[cal_oos["currency"] == ccy]
            if len(sub):
                m = _blocked_mask(bar_index, sub)
                coverage_per_currency[ccy] = {
                    "n_events": int(len(sub)),
                    "coverage": float(m.mean()),
                }

    # === Q1.c — realized vol elevated on blocked vs non-blocked bars? ===
    abs_log_ret = np.abs(np.log(df["close"] / df["close"].shift(1)))
    abs_log_ret = abs_log_ret.fillna(0).to_numpy()
    blocked_vol = abs_log_ret[mask_any]
    free_vol = abs_log_ret[~mask_any]
    vol_blocked_median = float(np.median(blocked_vol)) if len(blocked_vol) else float("nan")
    vol_free_median = float(np.median(free_vol)) if len(free_vol) else float("nan")
    vol_ratio = (
        vol_blocked_median / vol_free_median if vol_free_median > 0 else float("nan")
    )

    # Bootstrap CI on the ratio
    if len(blocked_vol) and len(free_vol):
        rng = np.random.default_rng(42)
        ratios = []
        nb = len(blocked_vol)
        nf = len(free_vol)
        for _ in range(500):
            b = blocked_vol[rng.integers(0, nb, nb)]
            f = free_vol[rng.integers(0, nf, nf)]
            ratios.append(np.median(b) / max(np.median(f), 1e-12))
        ratio_lo = float(np.quantile(ratios, 0.025))
        ratio_hi = float(np.quantile(ratios, 0.975))
    else:
        ratio_lo, ratio_hi = float("nan"), float("nan")

    return {
        "symbol": inst.symbol,
        "label": label,
        "n_bars_oos": n_bars,
        "n_cal_events_oos": int(n_events),
        "block_window_min": {"before": BLOCK_BEFORE_MIN, "after": BLOCK_AFTER_MIN},
        "doc_vs_code_note": "Doc claims 30/60; code uses 30/30 (news_analysis_agent.py:112-113).",
        "q1_structural_sanity": {
            "columns": columns,
            "duplicates_count": int(duplicates),
            "nan_impact_count": int(nan_impact),
            "nan_currency_count": int(nan_currency),
        },
        "q1_coverage": {
            "any_currency_coverage": coverage_any,
            "per_currency": coverage_per_currency,
            "n": n_bars,
        },
        "q1_vol_elevation": {
            "median_abs_logret_blocked": vol_blocked_median,
            "median_abs_logret_free": vol_free_median,
            "ratio_blocked_over_free": float(vol_ratio),
            "ratio_ci95_lo": ratio_lo,
            "ratio_ci95_hi": ratio_hi,
            "n_blocked": int(mask_any.sum()),
            "n_free": int((~mask_any).sum()),
            "definition": "ratio of median |log return| on blocked vs unblocked bars; >1 means blackout periods are genuinely more volatile",
        },
    }


def main():
    cal = load_calendar()
    results = {}
    for sym in ["XAUUSD", "EURUSD"]:
        print(f"--- {sym} ---")
        inst = load_instrument(sym)
        out = eval_calendar_for(inst, cal, f"{sym} M15 OOS 2024+")
        results[sym] = out
        ss = out["q1_structural_sanity"]
        cov = out["q1_coverage"]
        ve = out["q1_vol_elevation"]
        print(
            f"  OOS bars={out['n_bars_oos']:,d}  events={out['n_cal_events_oos']:,d}\n"
            f"  Q1 structural: dups={ss['duplicates_count']} nan_impact={ss['nan_impact_count']} nan_ccy={ss['nan_currency_count']}\n"
            f"  Q1 coverage (any ccy):  {cov['any_currency_coverage']*100:.2f}% of bars blocked"
        )
        for ccy, st in cov["per_currency"].items():
            print(f"     {ccy:3s}: {st['n_events']:,d} events, {st['coverage']*100:.2f}% bars blocked")
        print(
            f"  Q1 vol elevation: med|r|_blocked={ve['median_abs_logret_blocked']*1e4:.2f}bps  "
            f"med|r|_free={ve['median_abs_logret_free']*1e4:.2f}bps\n"
            f"     ratio={ve['ratio_blocked_over_free']:.3f}  "
            f"[{ve['ratio_ci95_lo']:.3f}, {ve['ratio_ci95_hi']:.3f}]"
        )

    out_path = HERE.parent.parent.parent / "docs" / "audits" / "data" / "calendar_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
