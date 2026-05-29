"""Evaluate Order Block (OB) on OOS data.

Prod definition:
  - Bullish OB at bar i: close[i-1] < open[i-1] (prev bearish) AND close[i] > open[i]
    (current bullish) AND high[i] > high[i-1]. Zone = [low[i-1], high[i-1]].
  - Bearish OB analogous.
  - OB_STRENGTH_NORM = zone_size / ATR (+ 0.2 FVG bonus if applicable).

Q1 (Justesse factuelle):
  - Inequality sanity: every flagged bar respects the pattern.
  - Zone sanity: BULLISH_OB_HIGH > BULLISH_OB_LOW (same for bearish).
  - Strength sanity: OB_STRENGTH_NORM = zone_size / ATR up to FVG bonus.

Q2 (Stabilité — fenêtre 16 bars):
  - Retest reaction: when price returns to the OB zone within 16 bars,
    does it bounce (close back outside in OB direction)? Reaction rate.
  - Never-touched rate: fraction of OBs that price never revisits within 16 bars.

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
    detect_ob_independent,
    f1_from_counts,
    load_instrument,
    verdict_f1,
)


LIFETIME_BARS = 16
MATCH_WINDOW = 5


def eval_ob_for(inst: InstrumentData, label: str) -> dict:
    df = inst.oos.copy()
    df["atr"] = df["ATR"].astype(float)
    df["ob_bull_high"] = df["BULLISH_OB_HIGH"]
    df["ob_bull_low"] = df["BULLISH_OB_LOW"]
    df["ob_bear_high"] = df["BEARISH_OB_HIGH"]
    df["ob_bear_low"] = df["BEARISH_OB_LOW"]
    df["ob_strength"] = df["OB_STRENGTH_NORM"].astype(float)

    bull_mask = df["ob_bull_high"].notna()
    bear_mask = df["ob_bear_high"].notna()
    n_bull = int(bull_mask.sum())
    n_bear = int(bear_mask.sum())
    n_events = n_bull + n_bear
    if n_events == 0:
        return {"symbol": inst.symbol, "n_events": 0, "skipped": True}

    # === Q1.a — pattern inequality sanity ===
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    bull = bull_mask.to_numpy()
    bear = bear_mask.to_numpy()
    n = len(df)

    bull_idx = np.where(bull)[0]
    bear_idx = np.where(bear)[0]
    pattern_ok = 0
    for i in bull_idx:
        if i == 0:
            continue
        if c[i - 1] < o[i - 1] and c[i] > o[i] and highs[i] > highs[i - 1]:
            pattern_ok += 1
    for i in bear_idx:
        if i == 0:
            continue
        if c[i - 1] > o[i - 1] and c[i] < o[i] and lows[i] < lows[i - 1]:
            pattern_ok += 1
    pattern_sanity = pattern_ok / n_events

    # === Q1.b — zone sanity: high > low ===
    bull_zone_ok = (df.loc[bull_mask, "ob_bull_high"] > df.loc[bull_mask, "ob_bull_low"]).sum()
    bear_zone_ok = (df.loc[bear_mask, "ob_bear_high"] > df.loc[bear_mask, "ob_bear_low"]).sum()
    zone_sanity = (bull_zone_ok + bear_zone_ok) / n_events

    # === Q1.c — strength sanity: matches zone_size/ATR (within 0.21 to allow FVG bonus) ===
    bull_size = df.loc[bull_mask, "ob_bull_high"] - df.loc[bull_mask, "ob_bull_low"]
    bear_size = df.loc[bear_mask, "ob_bear_high"] - df.loc[bear_mask, "ob_bear_low"]
    bull_expected = bull_size / df.loc[bull_mask, "atr"]
    bear_expected = bear_size / df.loc[bear_mask, "atr"]
    bull_diff = (df.loc[bull_mask, "ob_strength"] - bull_expected).abs()
    bear_diff = (df.loc[bear_mask, "ob_strength"] - bear_expected).abs()
    strength_ok = int((bull_diff <= 0.21).sum() + (bear_diff <= 0.21).sum())
    strength_sanity = strength_ok / n_events

    # === Q1.d — cross-method with impulse-OB reference (>=0.8xATR body) ===
    ref = detect_ob_independent(
        df[["open", "high", "low", "close", "ATR"]].rename(columns={"ATR": "atr"})
    )
    ref_bull = ref["ref_bull_ob"].to_numpy().astype(bool)
    ref_bear = ref["ref_bear_ob"].to_numpy().astype(bool)
    # Compute F1 separately then aggregate
    def _f1(prod_mask, ref_mask, window):
        matched_p = np.zeros(n, dtype=bool)
        matched_r = np.zeros(n, dtype=bool)
        for pi in np.where(prod_mask)[0]:
            lo_ = max(0, pi - window)
            hi_ = min(n - 1, pi + window)
            for ri in range(lo_, hi_ + 1):
                if ref_mask[ri] and not matched_r[ri]:
                    matched_p[pi] = True
                    matched_r[ri] = True
                    break
        tp = int(matched_p[prod_mask].sum())
        fp = int(prod_mask.sum() - tp)
        fn = int(ref_mask.sum() - matched_r.sum())
        p_, r_, f_ = f1_from_counts(tp, fp, fn)
        return tp, fp, fn, p_, r_, f_

    bull_tp, bull_fp, bull_fn, bull_p, bull_r, bull_f1 = _f1(bull, ref_bull, MATCH_WINDOW)
    bear_tp, bear_fp, bear_fn, bear_p, bear_r, bear_f1 = _f1(bear, ref_bear, MATCH_WINDOW)
    agg_tp = bull_tp + bear_tp
    agg_fp = bull_fp + bear_fp
    agg_fn = bull_fn + bear_fn
    agg_p, agg_r, agg_f1 = f1_from_counts(agg_tp, agg_fp, agg_fn)

    # === Q2 — reaction at retest within 16 bars ===
    reactions = 0
    never_touched = 0
    n_with_window = 0
    for is_bull, idx_arr in [(True, bull_idx), (False, bear_idx)]:
        for i in idx_arr:
            if is_bull:
                zone_low = df["ob_bull_low"].iloc[i]
                zone_high = df["ob_bull_high"].iloc[i]
            else:
                zone_low = df["ob_bear_low"].iloc[i]
                zone_high = df["ob_bear_high"].iloc[i]
            j_end = min(n, i + 1 + LIFETIME_BARS)
            future_h = highs[i + 1: j_end]
            future_l = lows[i + 1: j_end]
            future_c = c[i + 1: j_end]
            if len(future_h) == 0:
                continue
            n_with_window += 1
            touched = (future_l <= zone_high) & (future_h >= zone_low)
            if not touched.any():
                never_touched += 1
                continue
            # First touch index
            ti = int(touched.argmax())
            # Reaction: after first touch, does any subsequent close exit
            # the zone in the OB direction within the remaining lifetime?
            remaining_c = future_c[ti:]
            if is_bull:
                exits_up = remaining_c > zone_high
            else:
                exits_up = remaining_c < zone_low
            if exits_up.any():
                reactions += 1

    reaction_rate = reactions / n_with_window if n_with_window else float("nan")
    untouched_rate = never_touched / n_with_window if n_with_window else float("nan")
    react_arr = np.array(
        [1] * reactions + [0] * (n_with_window - reactions), dtype=float
    )
    _, react_lo, react_hi = bootstrap_ci(react_arr, np.mean, n_boot=1000) if n_with_window else (
        float("nan"), float("nan"), float("nan")
    )

    median_strength = float(df.loc[bull_mask | bear_mask, "ob_strength"].median())

    return {
        "symbol": inst.symbol,
        "label": label,
        "n_events": n_events,
        "n_bull": n_bull,
        "n_bear": n_bear,
        "median_strength_atr": median_strength,
        "q1_pattern_sanity": {
            "value": float(pattern_sanity),
            "definition": "engulfing-like pattern conditions satisfied on event bar",
            "n": n_events,
        },
        "q1_zone_sanity": {
            "value": float(zone_sanity),
            "definition": "OB_HIGH > OB_LOW",
            "n": n_events,
        },
        "q1_strength_sanity": {
            "value": float(strength_sanity),
            "definition": "OB_STRENGTH_NORM = (high-low)/ATR within 0.21 (allows FVG bonus 0.2)",
            "n": n_events,
        },
        "q1_cross_method_f1": {
            "f1_aggregated": float(agg_f1),
            "precision": float(agg_p),
            "recall": float(agg_r),
            "verdict": verdict_f1(agg_f1),
            "tp": agg_tp,
            "fp": agg_fp,
            "fn": agg_fn,
            "bull_f1": float(bull_f1),
            "bear_f1": float(bear_f1),
            "window": MATCH_WINDOW,
            "n_prod_events": n_events,
            "n_ref_events": int(ref_bull.sum() + ref_bear.sum()),
            "ref_method": "impulse OB (body > 0.8xATR + engulfing range)",
            "note": "Prod OB has no impulse filter; ref requires >=0.8xATR impulse. Low F1 reflects definition strictness mismatch.",
        },
        "q2_stability_16bars": {
            "reaction_rate_on_retest": float(reaction_rate),
            "reaction_rate_ci95_lo": float(react_lo),
            "reaction_rate_ci95_hi": float(react_hi),
            "never_touched_rate": float(untouched_rate),
            "n": n_with_window,
            "lifetime_bars": LIFETIME_BARS,
            "definition": "reaction_rate = fraction where, after first touch of the OB zone, a subsequent close exits in OB direction within 16 bars",
        },
    }


def main():
    results = {}
    for sym in ["XAUUSD", "EURUSD"]:
        print(f"--- {sym} ---")
        inst = load_instrument(sym)
        out = eval_ob_for(inst, f"{sym} M15 OOS 2024+")
        results[sym] = out
        if out.get("skipped"):
            print("  skipped (no events)")
            continue
        print(
            f"  events={out['n_events']:,d} bull={out['n_bull']:,d} bear={out['n_bear']:,d} "
            f"median strength={out['median_strength_atr']:.2f}xATR\n"
            f"  Q1 pattern sanity:    {out['q1_pattern_sanity']['value']:.4f}\n"
            f"  Q1 zone sanity:       {out['q1_zone_sanity']['value']:.4f}\n"
            f"  Q1 strength sanity:   {out['q1_strength_sanity']['value']:.4f}\n"
            f"  Q1 cross-method F1:   {out['q1_cross_method_f1']['f1_aggregated']:.3f} "
            f"P={out['q1_cross_method_f1']['precision']:.3f} "
            f"R={out['q1_cross_method_f1']['recall']:.3f} "
            f"{out['q1_cross_method_f1']['verdict']}  "
            f"(prod={out['q1_cross_method_f1']['n_prod_events']}, "
            f"ref={out['q1_cross_method_f1']['n_ref_events']})\n"
            f"  Q2 reaction-on-retest: {out['q2_stability_16bars']['reaction_rate_on_retest']:.4f}  "
            f"[{out['q2_stability_16bars']['reaction_rate_ci95_lo']:.3f}, "
            f"{out['q2_stability_16bars']['reaction_rate_ci95_hi']:.3f}]\n"
            f"     never-touched rate: {out['q2_stability_16bars']['never_touched_rate']:.4f}"
        )

    out_path = HERE.parent.parent.parent / "docs" / "audits" / "data" / "ob_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
