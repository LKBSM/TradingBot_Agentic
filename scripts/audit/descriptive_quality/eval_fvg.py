"""Evaluate FVG (Fair Value Gap) on OOS data.

Prod definition: bullish FVG at bar i iff low[i] > high[i-2] AND
(low[i] - high[i-2]) / ATR[i] > FVG_THRESHOLD (default 0.1xATR).

Q1 (Justesse factuelle):
  - Definitional sanity: every FVG_SIGNAL!=0 must satisfy the gap inequality.
  - Threshold sanity: |FVG_SIZE_NORM| > threshold when FVG_SIGNAL!=0.
  - Cross-method vs threshold-free textbook definition: compute precision/recall
    (prod is a strict subset of textbook by threshold cut — expect P=1.00).

Q2 (Stabilité):
  - Mitigation rate within K=16 bars: fraction of FVGs where price re-enters
    the gap zone (low<=FVG_high & high>=FVG_low) within 16 bars.
  - Median time to mitigation (bars).

Q3 (Calibration): N/A.
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
    bootstrap_ci,
    detect_fvg_independent,
    f1_from_counts,
    load_instrument,
    verdict_f1,
)


LIFETIME_BARS = 16
FVG_THRESHOLD = 0.1  # prod default


def eval_fvg_for(inst: InstrumentData, label: str) -> dict:
    df = inst.oos.copy()
    df["fvg_signal"] = df["FVG_SIGNAL"].astype(int)
    df["fvg_size"] = df["FVG_SIZE"].astype(float)
    df["fvg_size_norm"] = df["FVG_SIZE_NORM"].astype(float)
    df["fvg_dir"] = df["FVG_DIR"].astype(int)
    df["atr"] = df["ATR"].astype(float)

    sig_mask = df["fvg_signal"] != 0
    n_events = int(sig_mask.sum())
    if n_events == 0:
        return {"symbol": inst.symbol, "n_events": 0, "skipped": True}

    n_bull = int((df["fvg_signal"] == 1).sum())
    n_bear = int((df["fvg_signal"] == -1).sum())

    # === Q1.a — gap inequality must hold on every FVG_SIGNAL bar ===
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    sig = df["fvg_signal"].to_numpy()
    n = len(df)
    idx = np.where(sig != 0)[0]
    inequality_ok = 0
    for i in idx:
        if i < 2:
            continue
        if sig[i] == 1 and lows[i] > highs[i - 2]:
            inequality_ok += 1
        elif sig[i] == -1 and highs[i] < lows[i - 2]:
            inequality_ok += 1
    ineq_sanity = inequality_ok / n_events

    # === Q1.b — threshold sanity: |SIZE_NORM| > threshold ===
    sn = df.loc[sig_mask, "fvg_size_norm"].abs()
    thr_sanity = float((sn > FVG_THRESHOLD).sum()) / n_events

    # === Q1.c — cross-method vs textbook (threshold-free) ===
    ref = detect_fvg_independent(df[["open", "high", "low", "close", "ATR"]].rename(
        columns={"ATR": "atr"}
    ))
    ref_dir = ref["ref_fvg_dir"].to_numpy()
    # Same-bar match (FVG is a single-bar event by construction in both)
    prod_dir = sig
    matched_prod = (prod_dir != 0) & (prod_dir == ref_dir)
    tp = int(matched_prod.sum())
    fp = int(((prod_dir != 0) & (~matched_prod)).sum())
    fn = int(((ref_dir != 0) & (ref_dir != prod_dir)).sum())
    p, r, f1 = f1_from_counts(tp, fp, fn)
    n_ref = int((ref_dir != 0).sum())

    # === Q2 — mitigation rate within 16 bars ===
    # For each FVG, define the gap zone and check if any future bar's range touches it.
    mitigated = 0
    times = []
    n_with_window = 0
    for i in idx:
        if i < 2:
            continue
        j_end = min(n, i + 1 + LIFETIME_BARS)
        future_h = highs[i + 1: j_end]
        future_l = lows[i + 1: j_end]
        if len(future_h) == 0:
            continue
        n_with_window += 1
        if sig[i] == 1:
            zone_low = highs[i - 2]
            zone_high = lows[i]
        else:
            zone_low = highs[i]
            zone_high = lows[i - 2]
        # Mitigated iff any future bar overlaps the zone
        overlap = (future_l <= zone_high) & (future_h >= zone_low)
        if overlap.any():
            mitigated += 1
            times.append(int(overlap.argmax()) + 1)

    mit_rate = mitigated / n_with_window if n_with_window else float("nan")
    if times:
        median_time = float(np.median(times))
        _, mit_lo, mit_hi = bootstrap_ci(
            np.array([1] * mitigated + [0] * (n_with_window - mitigated), dtype=float),
            np.mean,
            n_boot=1000,
        )
    else:
        median_time = float("nan")
        mit_lo, mit_hi = float("nan"), float("nan")

    # Median FVG size (in ATRs) for context
    sizes_norm = df.loc[sig_mask, "fvg_size_norm"].abs().to_numpy()
    median_size_atr = float(np.median(sizes_norm))

    return {
        "symbol": inst.symbol,
        "label": label,
        "n_events": n_events,
        "n_bull": n_bull,
        "n_bear": n_bear,
        "median_size_atr": median_size_atr,
        "q1_inequality_sanity": {
            "value": float(ineq_sanity),
            "definition": "bullish FVG: low[i] > high[i-2]; bearish FVG: high[i] < low[i-2]",
            "n": n_events,
        },
        "q1_threshold_sanity": {
            "value": float(thr_sanity),
            "definition": f"|FVG_SIZE_NORM| > {FVG_THRESHOLD}",
            "n": n_events,
        },
        "q1_cross_method_f1": {
            "f1": float(f1),
            "precision": float(p),
            "recall": float(r),
            "verdict": verdict_f1(f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_prod_events": n_events,
            "n_ref_events": n_ref,
            "ref_method": "textbook FVG, threshold-free (any 3-bar gap)",
            "note": "Prod is a strict subset of ref by the ATR threshold. P=1 expected, R<1 reflects threshold cut.",
        },
        "q2_stability_16bars": {
            "mitigation_rate": float(mit_rate),
            "mitigation_rate_ci95_lo": float(mit_lo),
            "mitigation_rate_ci95_hi": float(mit_hi),
            "median_time_to_mitigation_bars": median_time,
            "n": n_with_window,
            "lifetime_bars": LIFETIME_BARS,
            "definition": "mitigation = any next-16 bar range overlaps the FVG zone",
        },
    }


def main():
    results = {}
    for sym in ["XAUUSD", "EURUSD"]:
        print(f"--- {sym} ---")
        inst = load_instrument(sym)
        out = eval_fvg_for(inst, f"{sym} M15 OOS 2024+")
        results[sym] = out
        if out.get("skipped"):
            print("  skipped (no events)")
            continue
        print(
            f"  events={out['n_events']:,d} bull={out['n_bull']:,d} bear={out['n_bear']:,d} "
            f"median size={out['median_size_atr']:.2f}xATR\n"
            f"  Q1 inequality sanity:   {out['q1_inequality_sanity']['value']:.4f}\n"
            f"  Q1 threshold sanity:    {out['q1_threshold_sanity']['value']:.4f}\n"
            f"  Q1 cross-method F1:     {out['q1_cross_method_f1']['f1']:.3f} "
            f"P={out['q1_cross_method_f1']['precision']:.3f} "
            f"R={out['q1_cross_method_f1']['recall']:.3f} "
            f"{out['q1_cross_method_f1']['verdict']}\n"
            f"  Q2 mitigation @16bars:  {out['q2_stability_16bars']['mitigation_rate']:.4f}  "
            f"[{out['q2_stability_16bars']['mitigation_rate_ci95_lo']:.3f}, "
            f"{out['q2_stability_16bars']['mitigation_rate_ci95_hi']:.3f}]\n"
            f"     median time:         {out['q2_stability_16bars']['median_time_to_mitigation_bars']} bars"
        )

    out_path = HERE.parent.parent.parent / "docs" / "audits" / "data" / "fvg_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
