"""MT-D1 Section 4 — open case trace. Engine READ-ONLY."""
import sys
from datetime import datetime, timezone
sys.path.insert(0, ".")
import numpy as np
from scripts.audit.mt_d1 import harness as H

candles = H.load_candles("H1")
enr = H.enrich(candles)
idx = list(enr.index)
c = enr["close"].values.astype(float)
h = enr["high"].values.astype(float)
l = enr["low"].values.astype(float)
uf = enr["UP_FRACTAL"].values.astype(float)
dfa = enr["DOWN_FRACTAL"].values.astype(float)

# Reproduce the engine loop while tracing current_low_structure through the dip.
n = len(c)
current_high_structure = h[0]; current_low_structure = l[0]
last_fractal_high = h[0]; last_fractal_low = l[0]
last_bos_up = -np.inf; last_bos_down = np.inf
bos_signal = np.zeros(n, int)

start = datetime(2026, 7, 24, 20, tzinfo=timezone.utc)
end = datetime(2026, 7, 28, 12, tzinfo=timezone.utc)

for i in range(min(50, n)):
    if not np.isnan(uf[i]):
        last_fractal_high = uf[i]; current_high_structure = max(current_high_structure, last_fractal_high)
    if not np.isnan(dfa[i]):
        last_fractal_low = dfa[i]; current_low_structure = min(current_low_structure, last_fractal_low)

print("bar_ts               close    struct_low  struct_high  trend  close<low?  new_fractal_low")
for i in range(1, n):
    if not np.isnan(uf[i]):
        last_fractal_high = uf[i]; current_high_structure = max(current_high_structure, last_fractal_high)
    nf_low = None
    if not np.isnan(dfa[i]):
        last_fractal_low = dfa[i]; current_low_structure = min(current_low_structure, last_fractal_low)
        nf_low = dfa[i]
    allow_up = last_fractal_high > last_bos_up
    allow_down = last_fractal_low < last_bos_down
    fired = 0
    if bos_signal[i-1] == -1 and c[i] > current_high_structure:
        current_low_structure = last_fractal_low; current_high_structure = last_fractal_high
        bos_signal[i] = 1; last_bos_up = c[i]; fired = 1
    elif bos_signal[i-1] == 1 and c[i] < current_low_structure:
        current_high_structure = last_fractal_high; current_low_structure = last_fractal_low
        bos_signal[i] = -1; last_bos_down = c[i]; fired = -1
    else:
        if bos_signal[i-1] >= 0 and c[i] > current_high_structure and allow_up:
            current_high_structure = last_fractal_high; current_low_structure = last_fractal_low
            bos_signal[i] = 1; last_bos_up = c[i]; fired = 1
        elif bos_signal[i-1] <= 0 and c[i] < current_low_structure and allow_down:
            current_low_structure = last_fractal_low; current_high_structure = last_fractal_high
            bos_signal[i] = -1; last_bos_down = c[i]; fired = -1
        else:
            bos_signal[i] = bos_signal[i-1]
    ts = idx[i]
    if start <= ts <= end:
        flag = "***CLOSE<LOW***" if c[i] < current_low_structure else ""
        ev = {1: "BOS/CHOCH UP", -1: "BOS/CHOCH DOWN", 0: ""}[fired]
        print(f"{ts}  {c[i]:8.2f}  {current_low_structure:8.2f}   {current_high_structure:8.2f}  "
              f"{bos_signal[i]:+d}   {flag:14s} {'' if nf_low is None else round(nf_low,2)}  {ev}")

# Also: lowest close in the dip vs the protected low at that time
print("\nSummary:")
mask = [(start <= t <= end) for t in idx]
sub = [(idx[i], c[i], l[i]) for i in range(n) if mask[i]]
mnc = min(sub, key=lambda x: x[1])
mnl = min(sub, key=lambda x: x[2])
print(f"  lowest CLOSE in window: {mnc[1]:.2f} at {mnc[0]}")
print(f"  lowest LOW   in window: {mnl[2]:.2f} at {mnl[0]}")
