"""Unit tests for the BOS retest state machine (strategy_features.calculate_bos_retest_fast).

State machine contract (per direction):
  IDLE (0) ─[BOS event with break_level L]─► AWAITING (+/-1)
  AWAITING ─[price touches L within retest_tol]─► ARMED (+/-2)
           ─[close past L by invalid_tol]────► IDLE (failed break)
           ─[awaiting_timeout bars elapse]────► IDLE (stale)
  ARMED    ─[armed_window bars elapse]────────► IDLE
           ─[close past L by invalid_tol]────► IDLE (support/resistance lost)
           ─[otherwise]──────────────────────► stays ARMED, emits retest_armed
  Any state ─[new BOS event]────────────────► state switches, regardless of direction
"""
import numpy as np
import pytest

from src.environment.strategy_features import (
    calculate_bos_retest_fast,
    _calculate_bos_retest_python,
)


def _run(
    closes, highs, lows, bos_event, bos_break_level, atr,
    retest_tol_atr=0.5, invalid_tol_atr=1.0,
    awaiting_timeout=20, armed_window=5,
):
    """Helper that runs the python fallback (deterministic, no JIT caching)."""
    return _calculate_bos_retest_python(
        np.asarray(closes, dtype=np.float64),
        np.asarray(highs, dtype=np.float64),
        np.asarray(lows, dtype=np.float64),
        np.asarray(bos_event, dtype=np.int32),
        np.asarray(bos_break_level, dtype=np.float64),
        np.asarray(atr, dtype=np.float64),
        retest_tol_atr, invalid_tol_atr,
        awaiting_timeout, armed_window,
    )


class TestBosRetestBasicArming:
    """After a BOS event, a pullback to the broken level should arm the setup."""

    def test_long_arms_after_pullback(self):
        # Bar 0: BOS_UP at break level 100.0, price closes at 101
        # Bars 1-2: price drifts up (no retest)
        # Bar 3: price dips to 100.2 (within 0.5 ATR = 0.5 of 100.0) → ARMED
        # Bars 4+: armed
        n = 7
        closes = [101, 101, 101, 100.5, 101, 101, 101]
        highs  = [101, 101.2, 101.2, 101, 101, 101, 101]
        lows   = [101, 101, 100.8, 100.2, 100.6, 100.8, 100.9]
        atr    = [1.0] * n
        bos_event = [1, 0, 0, 0, 0, 0, 0]
        bos_break_level = [100.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        state, armed = _run(closes, highs, lows, bos_event, bos_break_level, atr)

        # Event bar is AWAITING (+1)
        assert state[0] == 1
        assert armed[0] == 0
        # Bars 1-2 still awaiting
        assert state[1] == 1 and state[2] == 1
        # Bar 3: pullback hit → ARMED (+2)
        assert state[3] == 2
        assert armed[3] == 1
        # Bars 4-5 armed
        assert armed[4] == 1 and armed[5] == 1

    def test_short_arms_after_pullback(self):
        # Mirror of long case: BOS_DOWN, price rallies back to level
        n = 6
        closes = [99, 99, 99, 99.5, 99, 99]
        highs  = [99, 99.3, 99.3, 99.8, 99.4, 99.2]  # bar 3 high >= 99.5 (100 - 0.5)
        lows   = [99, 99, 99, 99, 99, 99]
        atr    = [1.0] * n
        bos_event = [-1, 0, 0, 0, 0, 0]
        bos_break_level = [100.0, np.nan, np.nan, np.nan, np.nan, np.nan]

        state, armed = _run(closes, highs, lows, bos_event, bos_break_level, atr)

        assert state[0] == -1
        # Bar 3: high 99.8 >= 100 - 0.5 → ARMED (-2)
        assert state[3] == -2
        assert armed[3] == -1


class TestBosRetestInvalidation:
    """Failed breaks (price closes past the level in the wrong direction) must void."""

    def test_long_invalidated_when_close_drops_below_level(self):
        # BOS_UP at break level 100.0, ATR=1.0, invalid_tol_atr=1.0
        # Bar 2: close 98.5 → 1.5 ATR below 100 → invalidated
        n = 4
        closes = [101, 101, 98.5, 98]
        highs  = [101, 101.2, 101, 99]
        lows   = [101, 100.9, 98, 97.5]
        atr    = [1.0] * n
        bos_event = [1, 0, 0, 0]
        bos_break_level = [100.0, np.nan, np.nan, np.nan]

        state, armed = _run(closes, highs, lows, bos_event, bos_break_level, atr)

        # Bar 2 invalidated → back to IDLE
        assert state[2] == 0
        # No armed bars at all
        assert armed.sum() == 0

    def test_long_armed_invalidated_when_support_breaks(self):
        # Arm on bar 1 (pullback), then on bar 3 close well below level → void
        n = 5
        closes = [101, 100.2, 100.3, 98.5, 98]
        highs  = [101, 100.5, 100.5, 100.0, 99]
        lows   = [101, 100.0, 100.1, 98, 97.5]
        atr    = [1.0] * n
        bos_event = [1, 0, 0, 0, 0]
        bos_break_level = [100.0, np.nan, np.nan, np.nan, np.nan]

        state, armed = _run(closes, highs, lows, bos_event, bos_break_level, atr)

        # Bar 1 pullback → armed
        assert state[1] == 2
        assert armed[1] == 1
        # Bar 3: close 98.5 is 1.5 ATR below level → voided
        assert state[3] == 0
        assert armed[3] == 0


class TestBosRetestTimeouts:
    """Awaiting and armed states must expire after their configured windows."""

    def test_awaiting_expires_after_timeout(self):
        # No pullback ever → after 3 bars (timeout=3), back to IDLE
        n = 6
        closes = [101, 102, 103, 104, 105, 106]
        highs  = [101, 102, 103, 104, 105, 106]
        lows   = [101, 102, 103, 104, 105, 106]  # never dips to 100.5
        atr    = [1.0] * n
        bos_event = [1, 0, 0, 0, 0, 0]
        bos_break_level = [100.0, np.nan, np.nan, np.nan, np.nan, np.nan]

        state, armed = _run(
            closes, highs, lows, bos_event, bos_break_level, atr,
            awaiting_timeout=3,
        )

        # Bars 1-3 still awaiting (bars_in counts 1,2,3 — all ≤ 3)
        assert state[1] == 1 and state[2] == 1 and state[3] == 1
        # Bar 4: bars_in=4 > timeout=3 → IDLE
        assert state[4] == 0
        assert armed.sum() == 0

    def test_armed_window_expires(self):
        # Arm on bar 1; armed_window=2 → armed for bars 1,2; bar 3 back to IDLE
        n = 6
        closes = [101, 100.3, 100.4, 100.5, 100.6, 100.7]
        highs  = [101, 100.5, 100.5, 100.7, 100.8, 100.9]
        lows   = [101, 100.1, 100.3, 100.4, 100.5, 100.6]
        atr    = [1.0] * n
        bos_event = [1, 0, 0, 0, 0, 0]
        bos_break_level = [100.0, np.nan, np.nan, np.nan, np.nan, np.nan]

        state, armed = _run(
            closes, highs, lows, bos_event, bos_break_level, atr,
            armed_window=2,
        )

        # Bar 1: armed
        assert state[1] == 2 and armed[1] == 1
        # Bar 2: still armed (bars_in=1 ≤ 2)
        assert state[2] == 2 and armed[2] == 1
        # Bar 3: bars_in=2 ≤ 2 → still armed
        assert state[3] == 2 and armed[3] == 1
        # Bar 4: bars_in=3 > 2 → IDLE
        assert state[4] == 0 and armed[4] == 0


class TestBosRetestOverride:
    """A new BOS event overrides any current state, regardless of direction."""

    def test_new_bos_event_resets_state(self):
        # Bar 0: BOS_UP awaiting
        # Bar 2: opposite BOS_DOWN event → state flips to AWAITING SHORT
        n = 4
        closes = [101, 101, 98, 98]
        highs  = [101, 101.5, 101, 99]
        lows   = [101, 100.5, 97, 97]
        atr    = [1.0] * n
        bos_event = [1, 0, -1, 0]
        bos_break_level = [100.0, np.nan, 99.0, np.nan]

        state, armed = _run(closes, highs, lows, bos_event, bos_break_level, atr)

        assert state[0] == 1
        # Bar 2: new BOS_DOWN event overrides
        assert state[2] == -1
        # No armed bars yet for the short
        assert armed[2] == 0


class TestBosRetestIntegration:
    """Smoke test: run against a realistic synthetic sequence end-to-end."""

    def test_full_cycle_long(self):
        """BOS_UP → pullback → armed → window expires, all with realistic ATR."""
        n = 10
        # Price breaks above 100, rallies, pulls back to 100.3, then drifts up
        closes = [100.5, 101.2, 102.0, 101.5, 100.3, 100.8, 101.2, 101.5, 101.8, 102.2]
        highs  = [100.7, 101.4, 102.3, 101.9, 100.6, 101.0, 101.4, 101.7, 102.0, 102.5]
        lows   = [100.2, 100.9, 101.5, 100.5, 100.1, 100.5, 100.9, 101.2, 101.5, 101.8]
        atr    = [0.6] * n
        bos_event = [1] + [0] * 9
        bos_break_level = [100.0] + [np.nan] * 9

        state, armed = _run(
            closes, highs, lows, bos_event, bos_break_level, atr,
            armed_window=3,
        )

        # Event bar
        assert state[0] == 1
        # Somewhere between bars 1-4 we should hit the retest (low 100.1 on bar 4 is well within tol=0.3)
        assert (armed[1:6] == 1).any(), "Expected at least one armed bar during pullback"
        # Retest fires on bar 4; with armed_window=3 it expires by bar 8
        assert state[-1] == 0, "Setup should have expired by bar 9 (armed_window=3)"


def test_numba_and_python_agree():
    """The fast path and python fallback must produce identical output."""
    np.random.seed(42)
    n = 200
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n)) * 0.3
    lows = closes - np.abs(np.random.randn(n)) * 0.3
    atr = np.full(n, 0.5)
    bos_event = np.zeros(n, dtype=np.int32)
    bos_break_level = np.full(n, np.nan)
    # Sprinkle a few events
    event_bars = [10, 50, 100, 150]
    for i, b in enumerate(event_bars):
        bos_event[b] = 1 if i % 2 == 0 else -1
        bos_break_level[b] = closes[b - 1]

    state_fast, armed_fast = calculate_bos_retest_fast(
        closes, highs, lows, bos_event, bos_break_level, atr,
    )
    state_py, armed_py = _calculate_bos_retest_python(
        closes, highs, lows, bos_event, bos_break_level, atr,
        retest_tol_atr=0.5, invalid_tol_atr=1.0,
        awaiting_timeout=20, armed_window=5,
    )

    np.testing.assert_array_equal(state_fast, state_py)
    np.testing.assert_array_equal(armed_fast, armed_py)
