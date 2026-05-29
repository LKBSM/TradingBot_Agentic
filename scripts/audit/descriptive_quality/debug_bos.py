"""Debug BOS level reality + cross-method to understand the low scores."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.audit.descriptive_quality._harness import (
    load_instrument, detect_bos_independent
)

inst = load_instrument("XAUUSD")
df = inst.oos.copy()

# 1) Level reality with wider lookback (500 bars) and wider tolerance (1, 2 x ATR)
events_idx = np.where(df["BOS_EVENT"].values != 0)[0]
print(f"Total OOS BOS events: {len(events_idx)}")
highs = df["high"].to_numpy()
lows = df["low"].to_numpy()
levels = df["BOS_BREAK_LEVEL"].to_numpy()
atrs = df["ATR"].to_numpy()
events = df["BOS_EVENT"].to_numpy()

for lookback in [200, 500, 1000]:
    for tol_atr in [0.1, 0.5, 1.0, 2.0]:
        n_ok = 0
        for i in events_idx:
            s = max(0, i - lookback)
            lvl = levels[i]
            atr = atrs[i]
            if np.isnan(lvl) or np.isnan(atr) or atr <= 0 or i <= s:
                continue
            tol = atr * tol_atr
            if events[i] == 1:
                past_max = highs[s:i].max()
                if abs(past_max - lvl) <= tol:
                    n_ok += 1
            else:
                past_min = lows[s:i].min()
                if abs(past_min - lvl) <= tol:
                    n_ok += 1
        rate = n_ok / len(events_idx)
        print(f"  lookback={lookback:4d}  tol={tol_atr:.2f}xATR  ->  {rate:.3f}")

# 2) Better level_reality: does the level appear ANYWHERE in past highs/lows (exact match)?
print("\nExact-match check (level == past high/low within 0.01xATR):")
n_match = 0
for i in events_idx:
    lvl = levels[i]
    atr = atrs[i]
    if np.isnan(lvl) or np.isnan(atr):
        continue
    tol = atr * 0.01
    if events[i] == 1:
        # was lvl the value of some past high?
        match = np.any(np.abs(highs[:i] - lvl) <= tol)
    else:
        match = np.any(np.abs(lows[:i] - lvl) <= tol)
    if match:
        n_match += 1
print(f"  exact past-bar OHLC match: {n_match/len(events_idx):.3f}")

# 3) Bound-check: is level inside [min, max] of recent 500 bars?
print("\nBound-check (level inside past 500-bar OHLC envelope):")
n_in = 0
for i in events_idx:
    s = max(0, i - 500)
    lvl = levels[i]
    if np.isnan(lvl) or i <= s:
        continue
    if lows[s:i].min() <= lvl <= highs[s:i].max():
        n_in += 1
print(f"  inside envelope: {n_in/len(events_idx):.3f}")

# 4) Cross-method with wider matching window
print("\nCross-method F1 vs window width:")
ref = detect_bos_independent(df[["open", "high", "low", "close", "ATR"]].rename(columns={"ATR": "atr"}))
ref_ev = ref["ref_bos_event"].to_numpy()
prod_ev = events
n = len(df)
for W in [2, 5, 10, 20]:
    matched_prod = np.zeros(n, dtype=bool)
    matched_ref = np.zeros(n, dtype=bool)
    for pi in np.where(prod_ev != 0)[0]:
        lo = max(0, pi - W)
        hi = min(n - 1, pi + W)
        for ri in range(lo, hi + 1):
            if ref_ev[ri] == prod_ev[pi] and not matched_ref[ri]:
                matched_prod[pi] = True
                matched_ref[ri] = True
                break
    tp = int(matched_prod[prod_ev != 0].sum())
    fp = int(((prod_ev != 0) & (~matched_prod)).sum())
    fn = int(((ref_ev != 0) & (~matched_ref)).sum())
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    print(f"  W={W:3d}  TP={tp:5d} FP={fp:5d} FN={fn:5d}  P={p:.3f} R={r:.3f} F1={f1:.3f}")

print(f"\nprod events: {(prod_ev != 0).sum()}, ref events: {(ref_ev != 0).sum()}")
