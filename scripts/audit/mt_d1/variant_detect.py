"""MT-D1 — instrumented + variant re-implementation of the BOS/CHOCH loop.

This is a STANDALONE copy used ONLY for the offline audit report. The repo
engine (`src/environment/strategy_features.py`) is NEVER modified. The default
mode here is byte-for-byte the engine's Python fallback
(`_calculate_bos_choch_python`); a parity check in run_audit.py asserts the
default variant reproduces the real engine's event stream exactly, so the
"variations" are provably deltas off the true rule.

Two purposes:
  1. Instrumentation — for every bar, record the structural state and, when a
     prior swing level is crossed but NO event fires, WHICH condition blocked it
     (section 3B non-events).
  2. Variants (section 7/8), each toggled independently, engine untouched:
       * cross_mode = "close" (engine) | "wick"  — break by close vs by high/low
       * margin_atr  = 0.0 (engine) | k          — require close beyond level by
                       k×ATR before it counts as a break
     FRACTAL_WINDOW is NOT varied here — that is a real SMCConfig knob and is
     swept through the actual engine in run_audit.py.
"""
from __future__ import annotations

import numpy as np


def detect(
    closes, highs, lows, up_fractals, down_fractals, atr=None,
    *, cross_mode="close", margin_atr=0.0,
):
    """Return dict with per-bar arrays: bos_event, choch_signal, bos_signal,
    bos_break_level, and a list of `blocked` crossings (non-events).

    Mirrors `_calculate_bos_choch_python` exactly when cross_mode='close' and
    margin_atr=0. `atr` only needed when margin_atr>0.
    """
    n = len(closes)
    bos_signal = np.zeros(n, dtype=np.int32)
    choch_signal = np.zeros(n, dtype=np.int32)
    bos_event = np.zeros(n, dtype=np.int32)
    bos_break_level = np.full(n, np.nan, dtype=np.float64)

    current_high_structure = highs[0]
    current_low_structure = lows[0]
    last_fractal_high = highs[0]
    last_fractal_low = lows[0]
    last_bos_up_level = -np.inf
    last_bos_down_level = np.inf

    blocked = []  # crossings that did NOT produce an event, with the reason

    def up_probe(i):
        return highs[i] if cross_mode == "wick" else closes[i]

    def down_probe(i):
        return lows[i] if cross_mode == "wick" else closes[i]

    def m(i):
        if margin_atr <= 0 or atr is None or np.isnan(atr[i]):
            return 0.0
        return margin_atr * atr[i]

    for i in range(min(50, n)):
        if not np.isnan(up_fractals[i]):
            last_fractal_high = up_fractals[i]
            current_high_structure = max(current_high_structure, last_fractal_high)
        if not np.isnan(down_fractals[i]):
            last_fractal_low = down_fractals[i]
            current_low_structure = min(current_low_structure, last_fractal_low)

    for i in range(1, n):
        if not np.isnan(up_fractals[i]):
            last_fractal_high = up_fractals[i]
            current_high_structure = max(current_high_structure, last_fractal_high)
        if not np.isnan(down_fractals[i]):
            last_fractal_low = down_fractals[i]
            current_low_structure = min(current_low_structure, last_fractal_low)

        allow_bos_up = last_fractal_high > last_bos_up_level
        allow_bos_down = last_fractal_low < last_bos_down_level
        up_p = up_probe(i)
        dn_p = down_probe(i)
        marg = m(i)

        fired = False
        if bos_signal[i - 1] == -1 and up_p > current_high_structure + marg:
            choch_signal[i] = 1
            bos_break_level[i] = current_high_structure
            current_low_structure = last_fractal_low
            current_high_structure = last_fractal_high
            bos_signal[i] = 1
            bos_event[i] = 1
            last_bos_up_level = closes[i]
            fired = True
        elif bos_signal[i - 1] == 1 and dn_p < current_low_structure - marg:
            choch_signal[i] = -1
            bos_break_level[i] = current_low_structure
            current_high_structure = last_fractal_high
            current_low_structure = last_fractal_low
            bos_signal[i] = -1
            bos_event[i] = -1
            last_bos_down_level = closes[i]
            fired = True
        elif choch_signal[i] == 0:
            if bos_signal[i - 1] >= 0 and up_p > current_high_structure + marg and allow_bos_up:
                bos_break_level[i] = current_high_structure
                bos_signal[i] = 1
                bos_event[i] = 1
                current_high_structure = last_fractal_high
                current_low_structure = last_fractal_low
                last_bos_up_level = closes[i]
                fired = True
            elif bos_signal[i - 1] <= 0 and dn_p < current_low_structure - marg and allow_bos_down:
                bos_break_level[i] = current_low_structure
                bos_signal[i] = -1
                bos_event[i] = -1
                current_low_structure = last_fractal_low
                current_high_structure = last_fractal_high
                last_bos_down_level = closes[i]
                fired = True
            else:
                bos_signal[i] = bos_signal[i - 1]

        # ---- Non-event diagnostics (default engine mode only) ----------------
        # A "swing crossing without event": the bar's HIGH pierced the structural
        # high (resp. LOW pierced structural low) but no BOS/CHOCH fired here.
        if not fired:
            if highs[i] > current_high_structure:
                reason = None
                if closes[i] <= current_high_structure:
                    reason = "wick_only_close_inside"       # crossed by wick, close did not confirm
                elif not (last_fractal_high > last_bos_up_level):
                    reason = "level_already_broken"         # no new higher swing since last up-break
                if reason:
                    blocked.append({
                        "k": i, "side": "up", "reason": reason,
                        "structural_high": float(current_high_structure),
                        "bar_high": float(highs[i]), "bar_close": float(closes[i]),
                    })
            if lows[i] < current_low_structure:
                reason = None
                if closes[i] >= current_low_structure:
                    reason = "wick_only_close_inside"
                elif not (last_fractal_low < last_bos_down_level):
                    reason = "level_already_broken"
                if reason:
                    blocked.append({
                        "k": i, "side": "down", "reason": reason,
                        "structural_low": float(current_low_structure),
                        "bar_low": float(lows[i]), "bar_close": float(closes[i]),
                    })

    return {
        "bos_event": bos_event,
        "choch_signal": choch_signal,
        "bos_signal": bos_signal,
        "bos_break_level": bos_break_level,
        "blocked": blocked,
    }


def count_events(res):
    be = res["bos_event"]
    ch = res["choch_signal"]
    return int((be != 0).sum()), int((ch != 0).sum())
