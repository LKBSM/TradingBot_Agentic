"""Evaluate CHOCH (Change of Character) on OOS data.

CHOCH is, per the engine, a BOS event that occurs in the opposite direction
of the previous bos_signal — i.e., a structural trend reversal. The doc
exposes CHOCH_SIGNAL with the same break-level metadata as BOS.

Q1 (Justesse factuelle):
  - Definitional sanity: every CHOCH_SIGNAL!=0 must coincide with a
    BOS_EVENT of the same sign on the same bar.
  - Reversal sanity: CHOCH+1 must come after a prior bos_signal=-1 stretch.
  - Cross-method: how many prod CHOCHs match an opposite-sign event from
    the independent 3-bar swing detector within ±5 bars?

Q2 (Stabilité — fenêtre 16 bars):
  - The headline CHOCH claim is "trend reversal." Test: after a bull CHOCH,
    does the next 16-bar mean close exceed the broken level? Mean hold rate.
  - no_immediate_reversal: fraction of CHOCHs not followed by an opposite
    CHOCH within 16 bars (else the "reversal" was noise).

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
    detect_bos_independent,
    f1_from_counts,
    load_instrument,
    verdict_f1,
)


LIFETIME_BARS = 16
MATCH_WINDOW = 5


def eval_choch_for(inst: InstrumentData, label: str) -> dict:
    df = inst.oos.copy()
    df["bos_event"] = df["BOS_EVENT"].astype(int)
    df["bos_signal"] = df["BOS_SIGNAL"].astype(int)
    df["choch_signal"] = df["CHOCH_SIGNAL"].astype(int)
    df["bos_level"] = df["BOS_BREAK_LEVEL"].astype(float)

    choch_mask = df["choch_signal"] != 0
    n_events = int(choch_mask.sum())
    if n_events == 0:
        return {"symbol": inst.symbol, "n_events": 0, "skipped": True}

    n_bull = int((df["choch_signal"] == 1).sum())
    n_bear = int((df["choch_signal"] == -1).sum())

    # === Q1.a — definitional sanity: every CHOCH should coincide with a same-sign BOS_EVENT ===
    same_sign = ((df["choch_signal"] != 0) & (df["choch_signal"] == df["bos_event"])).sum()
    def_sanity = float(same_sign) / n_events

    # === Q1.b — reversal sanity: prior bos_signal should be opposite ===
    # Look at bos_signal[i-1] for each choch at i; should equal -choch_signal[i]
    choch = df["choch_signal"].to_numpy()
    bsig = df["bos_signal"].to_numpy()
    idx = np.where(choch != 0)[0]
    reversal_ok = 0
    for i in idx:
        if i == 0:
            continue
        # Find the last non-zero bos_signal before bar i (since choch sets bos_signal at i too)
        prev = 0
        for j in range(i - 1, max(-1, i - 200), -1):
            if bsig[j] != 0:
                prev = bsig[j]
                break
        if prev != 0 and prev == -choch[i]:
            reversal_ok += 1
    reversal_sanity = reversal_ok / n_events

    # === Q1.c — cross-method match with independent reference ===
    # An independent CHOCH is: a sign change in the running ref bos_signal-equivalent.
    ref = detect_bos_independent(
        df[["open", "high", "low", "close", "ATR"]].rename(columns={"ATR": "atr"})
    )
    ref_ev = ref["ref_bos_event"].to_numpy()
    # Detect ref CHOCH: events whose sign differs from previous ref event
    ref_choch = np.zeros(len(ref_ev), dtype=np.int32)
    prev_sign = 0
    for i in range(len(ref_ev)):
        if ref_ev[i] != 0:
            if prev_sign != 0 and ref_ev[i] != prev_sign:
                ref_choch[i] = ref_ev[i]
            prev_sign = ref_ev[i]
    n_ref = int((ref_choch != 0).sum())

    # Match within ±MATCH_WINDOW
    n = len(df)
    matched_prod = np.zeros(n, dtype=bool)
    matched_ref = np.zeros(n, dtype=bool)
    for pi in idx:
        lo = max(0, pi - MATCH_WINDOW)
        hi = min(n - 1, pi + MATCH_WINDOW)
        for ri in range(lo, hi + 1):
            if ref_choch[ri] == choch[pi] and not matched_ref[ri]:
                matched_prod[pi] = True
                matched_ref[ri] = True
                break
    tp = int(matched_prod[choch != 0].sum())
    fp = int(((choch != 0) & (~matched_prod)).sum())
    fn = int(((ref_choch != 0) & (~matched_ref)).sum())
    precision, recall, f1 = f1_from_counts(tp, fp, fn)

    # === Q2 — stability of the new direction ===
    closes = df["close"].to_numpy()
    levels = df["bos_level"].to_numpy()
    hold_fracs = []
    no_immediate_rev = 0
    n_with_window = 0
    for i in idx:
        lvl = levels[i]
        d = choch[i]
        if np.isnan(lvl):
            continue
        j_end = min(n, i + 1 + LIFETIME_BARS)
        future_close = closes[i + 1: j_end]
        future_choch = choch[i + 1: j_end]
        if len(future_close) == 0:
            continue
        n_with_window += 1
        if d == 1:
            respects = future_close >= lvl
        else:
            respects = future_close <= lvl
        hold_fracs.append(respects.mean())
        opp = future_choch == -d
        if not opp.any():
            no_immediate_rev += 1

    hold_arr = np.array(hold_fracs)
    hold_mean = float(hold_arr.mean()) if len(hold_arr) else float("nan")
    if len(hold_arr):
        _, hold_lo, hold_hi = bootstrap_ci(hold_arr, np.mean, n_boot=1000)
    else:
        hold_lo, hold_hi = float("nan"), float("nan")
    no_imm_rate = no_immediate_rev / n_with_window if n_with_window else float("nan")

    return {
        "symbol": inst.symbol,
        "label": label,
        "n_events": n_events,
        "n_bull": n_bull,
        "n_bear": n_bear,
        "q1_definitional_sanity": {
            "value": float(def_sanity),
            "definition": "CHOCH_SIGNAL!=0 must coincide with same-sign BOS_EVENT",
            "n": n_events,
        },
        "q1_reversal_sanity": {
            "value": float(reversal_sanity),
            "definition": "prior bos_signal must be of opposite sign (trend reversal)",
            "n": n_events,
        },
        "q1_cross_method_f1": {
            "window": MATCH_WINDOW,
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "verdict": verdict_f1(f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_prod_events": n_events,
            "n_ref_events": n_ref,
            "ref_method": "3-bar swing CHOCH (sign change)",
        },
        "q2_stability_16bars": {
            "hold_rate_mean": hold_mean,
            "hold_rate_ci95_lo": float(hold_lo),
            "hold_rate_ci95_hi": float(hold_hi),
            "no_immediate_reversal_rate": float(no_imm_rate),
            "n": n_with_window,
            "lifetime_bars": LIFETIME_BARS,
            "definition": "hold_rate_mean = fraction of next-16 closes respecting CHOCH direction; no_immediate_reversal_rate = fraction with no opposite CHOCH in 16 bars",
        },
    }


def main():
    results = {}
    for sym in ["XAUUSD", "EURUSD"]:
        print(f"--- {sym} ---")
        inst = load_instrument(sym)
        out = eval_choch_for(inst, f"{sym} M15 OOS 2024+")
        results[sym] = out
        if out.get("skipped"):
            print("  skipped (no events)")
            continue
        q1d = out["q1_definitional_sanity"]
        q1r = out["q1_reversal_sanity"]
        q1f1 = out["q1_cross_method_f1"]
        q2 = out["q2_stability_16bars"]
        print(
            f"  events={out['n_events']:,d} bull={out['n_bull']:,d} bear={out['n_bear']:,d}\n"
            f"  Q1 def sanity (CHOCH==BOS):  {q1d['value']:.4f}\n"
            f"  Q1 reversal sanity:          {q1r['value']:.4f}\n"
            f"  Q1 cross-method F1 W=5:      {q1f1['f1']:.3f}  P={q1f1['precision']:.3f} "
            f"R={q1f1['recall']:.3f} {q1f1['verdict']}  (prod={q1f1['n_prod_events']}, ref={q1f1['n_ref_events']})\n"
            f"  Q2 hold-rate @16bars:        {q2['hold_rate_mean']:.4f}  "
            f"[{q2['hold_rate_ci95_lo']:.3f}, {q2['hold_rate_ci95_hi']:.3f}]\n"
            f"     no-immediate-reversal:    {q2['no_immediate_reversal_rate']:.4f}"
        )

    out_path = HERE.parent.parent.parent / "docs" / "audits" / "data" / "choch_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
