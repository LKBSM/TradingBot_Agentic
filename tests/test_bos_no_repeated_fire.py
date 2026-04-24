"""Regression: BOS_EVENT must not fire every bar in a trend.

The replay harness previously showed `bars_with_bos == bars_processed`
(100% firing) on a 20k-bar XAUUSD replay. Root cause: after a continuation
BOS, the structure was reset to `last_fractal_high`, which was already
below the just-broken level — so the next bar's close exceeded it too and
BOS fired again. Every bar in an uptrend produced a BOS event.

Fix: after a BOS event in a direction, suppress further BOS events in
that direction until a NEW fractal forms above (below) the breakout
level. This enforces canonical SMC structure: break → pullback → new
swing → break.
"""

import numpy as np

from src.environment.strategy_features import calculate_bos_choch_fast


def _make_fractals_from_highs_lows(highs: np.ndarray, lows: np.ndarray, window: int = 2):
    n = len(highs)
    up = np.full(n, np.nan)
    down = np.full(n, np.nan)
    for i in range(window, n - window):
        hi_window = highs[i - window : i + window + 1]
        lo_window = lows[i - window : i + window + 1]
        if highs[i] == hi_window.max():
            up[i] = highs[i]
        if lows[i] == lo_window.min():
            down[i] = lows[i]
    return up, down


class TestBOSDoesNotFireEveryBar:
    def test_staircase_uptrend_gets_sparse_bos_events(self):
        np.random.seed(42)
        n = 200
        closes = np.zeros(n)
        highs = np.zeros(n)
        lows = np.zeros(n)
        price = 100.0
        for i in range(n):
            cycle = i % 13
            if cycle < 10:
                price += 0.8
            else:
                price -= 0.6
            price += np.random.randn() * 0.1
            closes[i] = price
            highs[i] = price + 0.3
            lows[i] = price - 0.3

        up, down = _make_fractals_from_highs_lows(highs, lows)
        bos_sig, choch_sig, bos_evt, _brk = calculate_bos_choch_fast(closes, highs, lows, up, down)

        event_count = int((bos_evt != 0).sum())
        event_rate = event_count / n
        assert event_rate < 0.15, (
            f"BOS_EVENT fired on {event_rate:.1%} of bars — regression of "
            f"the 100%-firing bug. Expected <15%."
        )
        assert event_count >= 3, (
            f"Only {event_count} events on a 200-bar staircase uptrend — "
            "BOS detector may now be over-suppressed."
        )

    def test_pure_linear_uptrend_emits_single_bos(self):
        n = 100
        closes = np.linspace(100, 200, n)
        highs = closes + 0.1
        lows = closes - 0.1

        up, down = _make_fractals_from_highs_lows(highs, lows)
        bos_sig, choch_sig, bos_evt, _brk = calculate_bos_choch_fast(closes, highs, lows, up, down)

        event_count = int((bos_evt != 0).sum())
        assert event_count <= 2, (
            f"Pure linear uptrend produced {event_count} BOS events. With no "
            "pullbacks there are no new fractals, so only the initial break "
            "should fire."
        )

    def test_downtrend_bos_also_sparse(self):
        np.random.seed(7)
        n = 200
        closes = np.zeros(n)
        highs = np.zeros(n)
        lows = np.zeros(n)
        price = 200.0
        for i in range(n):
            cycle = i % 13
            if cycle < 10:
                price -= 0.8
            else:
                price += 0.6
            price += np.random.randn() * 0.1
            closes[i] = price
            highs[i] = price + 0.3
            lows[i] = price - 0.3

        up, down = _make_fractals_from_highs_lows(highs, lows)
        bos_sig, choch_sig, bos_evt, _brk = calculate_bos_choch_fast(closes, highs, lows, up, down)

        event_rate = (bos_evt != 0).sum() / n
        assert event_rate < 0.15, (
            f"Downtrend BOS_EVENT rate {event_rate:.1%} — symmetric bug "
            "exists on short side."
        )
