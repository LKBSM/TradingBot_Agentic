"""Evaluate the BOS Retest State Machine on OOS data.

State semantics (per strategy_features.py:259-334):
  0  idle
  ±1 AWAITING retest (BOS just fired, looking for pullback to level)
  ±2 ARMED (retest occurred, setup live until invalidation or armed_window)

Q1 (Justesse factuelle):
  - Definitional sanity: every state transition AWAITING→ARMED must
    correspond to low <= level + 0.5xATR (bull) / high >= level - 0.5xATR (bear).
  - Conversion rate: fraction of BOS events that subsequently reach ARMED.
  - Time-to-arm: median bars between BOS event and ARMED transition.

Q2 (Stabilité):
  - ARMED outcomes within armed_window (30 bars): continuation (close passes
    level + 1xATR in BOS direction) vs invalidation (close past level - 1xATR
    against direction) vs timeout.
  - retest_armed bars distribution.

Q3 (Calibration): N/A — no probabilistic claim.
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
    load_instrument,
)


RETEST_TOL_ATR = 0.5
INVALID_TOL_ATR = 1.0
AWAITING_TIMEOUT = 20
ARMED_WINDOW = 30


def eval_retest_for(inst: InstrumentData, label: str) -> dict:
    df = inst.oos.copy()
    df["state"] = df["BOS_RETEST_STATE"].astype(int)
    df["armed_flag"] = df["BOS_RETEST_ARMED"].astype(int)
    df["bos_event"] = df["BOS_EVENT"].astype(int)
    df["bos_level"] = df["BOS_BREAK_LEVEL"].astype(float)
    df["atr"] = df["ATR"].astype(float)

    state = df["state"].to_numpy()
    armed = df["armed_flag"].to_numpy()
    bos_ev = df["bos_event"].to_numpy()
    bos_lvl = df["bos_level"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atrs = df["atr"].to_numpy()
    n = len(df)

    # === Track each BOS event lifecycle ===
    bos_indices = np.where(bos_ev != 0)[0]
    n_bos = len(bos_indices)
    if n_bos == 0:
        return {"symbol": inst.symbol, "n_bos": 0, "skipped": True}

    conversions = 0  # BOS events that reach ARMED state
    times_to_arm = []
    outcomes = {"continuation": 0, "invalidation": 0, "timeout": 0}
    arm_touch_ok = 0
    arm_transitions = 0
    n_assessed = 0

    for ev_i in bos_indices:
        d = bos_ev[ev_i]
        lvl = bos_lvl[ev_i]
        if np.isnan(lvl):
            continue
        # Search forward for ARMED transition (state hits ±2 with same sign)
        next_i = min(n, ev_i + 1 + AWAITING_TIMEOUT + 5)
        sub_state = state[ev_i + 1: next_i]
        if d == 1:
            arm_idx_rel = np.where(sub_state == 2)[0]
        else:
            arm_idx_rel = np.where(sub_state == -2)[0]
        if len(arm_idx_rel) == 0:
            outcomes["timeout"] += 1
            n_assessed += 1
            continue
        arm_i_rel = int(arm_idx_rel[0])
        arm_i_abs = ev_i + 1 + arm_i_rel
        conversions += 1
        times_to_arm.append(arm_i_rel + 1)
        arm_transitions += 1
        a = atrs[arm_i_abs] if not np.isnan(atrs[arm_i_abs]) else 0.0
        # Definitional sanity for the AWAITING→ARMED touch
        if d == 1 and lows[arm_i_abs] <= lvl + RETEST_TOL_ATR * a:
            arm_touch_ok += 1
        elif d == -1 and highs[arm_i_abs] >= lvl - RETEST_TOL_ATR * a:
            arm_touch_ok += 1
        # Outcome within ARMED_WINDOW
        end_i = min(n, arm_i_abs + 1 + ARMED_WINDOW)
        win = closes[arm_i_abs + 1: end_i]
        win_atr = atrs[arm_i_abs + 1: end_i]
        if len(win) == 0:
            outcomes["timeout"] += 1
            n_assessed += 1
            continue
        win_atr_safe = np.where(np.isnan(win_atr) | (win_atr <= 0), a, win_atr)
        if d == 1:
            cont_mask = win >= lvl + INVALID_TOL_ATR * win_atr_safe
            inv_mask = win <= lvl - INVALID_TOL_ATR * win_atr_safe
        else:
            cont_mask = win <= lvl - INVALID_TOL_ATR * win_atr_safe
            inv_mask = win >= lvl + INVALID_TOL_ATR * win_atr_safe
        first_cont = cont_mask.argmax() if cont_mask.any() else 99999
        first_inv = inv_mask.argmax() if inv_mask.any() else 99999
        if first_cont < first_inv:
            outcomes["continuation"] += 1
        elif first_inv < first_cont:
            outcomes["invalidation"] += 1
        else:
            outcomes["timeout"] += 1
        n_assessed += 1

    conv_rate = conversions / n_assessed if n_assessed else float("nan")
    arm_def_sanity = arm_touch_ok / arm_transitions if arm_transitions else float("nan")
    cont_rate = outcomes["continuation"] / n_assessed if n_assessed else float("nan")
    inv_rate = outcomes["invalidation"] / n_assessed if n_assessed else float("nan")
    timeout_rate = outcomes["timeout"] / n_assessed if n_assessed else float("nan")

    cont_arr = np.array(
        [1] * outcomes["continuation"]
        + [0] * (n_assessed - outcomes["continuation"]),
        dtype=float,
    )
    _, cont_lo, cont_hi = bootstrap_ci(cont_arr, np.mean, n_boot=1000) if n_assessed else (
        float("nan"), float("nan"), float("nan")
    )

    median_time_to_arm = float(np.median(times_to_arm)) if times_to_arm else float("nan")

    # Also: how many bars are in ARMED state (descriptive global)
    armed_bars = int(np.abs(armed).sum())
    armed_density = armed_bars / n

    return {
        "symbol": inst.symbol,
        "label": label,
        "n_bos": n_bos,
        "n_assessed": n_assessed,
        "q1_definitional_sanity": {
            "value": float(arm_def_sanity),
            "definition": f"on AWAITING->ARMED, bull: low <= level + {RETEST_TOL_ATR}xATR; bear: high >= level - {RETEST_TOL_ATR}xATR",
            "n_transitions": arm_transitions,
        },
        "q1_conversion_rate": {
            "value": float(conv_rate),
            "definition": "fraction of BOS events that reach ARMED (retest occurred) within awaiting_timeout",
            "n": n_assessed,
            "awaiting_timeout_bars": AWAITING_TIMEOUT,
        },
        "q1_time_to_arm": {
            "median_bars": median_time_to_arm,
            "n_armed": conversions,
        },
        "q2_armed_outcomes": {
            "continuation_rate": float(cont_rate),
            "continuation_ci95_lo": float(cont_lo),
            "continuation_ci95_hi": float(cont_hi),
            "invalidation_rate": float(inv_rate),
            "timeout_rate": float(timeout_rate),
            "n": n_assessed,
            "armed_window_bars": ARMED_WINDOW,
            "invalid_tol_atr": INVALID_TOL_ATR,
            "definition": "outcome based on first close past level ± 1xATR in either direction within armed_window; ties or no-move = timeout",
        },
        "armed_density": {
            "armed_bars": armed_bars,
            "total_bars": n,
            "density": float(armed_density),
        },
    }


def main():
    results = {}
    for sym in ["XAUUSD", "EURUSD"]:
        print(f"--- {sym} ---")
        inst = load_instrument(sym)
        out = eval_retest_for(inst, f"{sym} M15 OOS 2024+")
        results[sym] = out
        if out.get("skipped"):
            print("  skipped (no BOS events)")
            continue
        print(
            f"  BOS events={out['n_bos']:,d}  assessed={out['n_assessed']:,d}\n"
            f"  Q1 def sanity (touch ok):   {out['q1_definitional_sanity']['value']:.4f}\n"
            f"  Q1 conversion rate:         {out['q1_conversion_rate']['value']:.4f}\n"
            f"     median time-to-arm:      {out['q1_time_to_arm']['median_bars']} bars\n"
            f"  Q2 ARMED outcomes (30bars):\n"
            f"     continuation: {out['q2_armed_outcomes']['continuation_rate']:.4f}  "
            f"[{out['q2_armed_outcomes']['continuation_ci95_lo']:.3f}, "
            f"{out['q2_armed_outcomes']['continuation_ci95_hi']:.3f}]\n"
            f"     invalidation: {out['q2_armed_outcomes']['invalidation_rate']:.4f}\n"
            f"     timeout:      {out['q2_armed_outcomes']['timeout_rate']:.4f}\n"
            f"  armed density: {out['armed_density']['density']*100:.2f}% of bars"
        )

    out_path = HERE.parent.parent.parent / "docs" / "audits" / "data" / "retest_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
