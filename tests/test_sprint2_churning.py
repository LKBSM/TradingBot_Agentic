# =============================================================================
# Sprint 2 Validation: Remove Churning Exploit (RW-1)
# =============================================================================
# Verifies that the unconditional open_bonus=0.3 has been replaced with a
# deferred entry bonus that only fires after MIN_HOLD_FOR_BONUS bars.
#
# Run with: python -m pytest tests/test_sprint2_churning.py -v
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    ACTION_HOLD, ACTION_OPEN_LONG, ACTION_CLOSE_LONG,
    ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT,
    POSITION_FLAT, POSITION_LONG, POSITION_SHORT,
    MIN_HOLD_FOR_BONUS,
)
from src.environment.environment import TradingEnv


def _make_data(n_rows=800, base_price=2000.0):
    """Create flat test data (no trend) for controlled reward testing."""
    np.random.seed(42)
    prices = np.full(n_rows, base_price) + np.random.normal(0, 0.5, n_rows)

    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=n_rows, freq='15min'),
        'Open': prices * 0.999,
        'High': prices * 1.002,
        'Low': prices * 0.998,
        'Close': prices,
        'Volume': np.full(n_rows, 500),
        'ATR': np.full(n_rows, 10.0),
        'RSI': np.full(n_rows, 50.0),
        'BOS_SIGNAL': np.zeros(n_rows),
        'OB_SIGNAL': np.zeros(n_rows),
    })
    df.set_index('Date', inplace=True)
    return df


def _make_env():
    df = _make_data()
    return TradingEnv(df, strict_scaler_mode=False)


def _step_n(env, action, n):
    """Step environment n times with given action, return last result."""
    result = None
    for _ in range(n):
        result = env.step(action)
        if result[2]:  # done
            break
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Config constant exists
# ─────────────────────────────────────────────────────────────────────────────
def test_min_hold_for_bonus_exists():
    """MIN_HOLD_FOR_BONUS is defined in config.py."""
    assert isinstance(MIN_HOLD_FOR_BONUS, int)
    assert MIN_HOLD_FOR_BONUS >= 1, "Must require at least 1 bar hold"
    assert MIN_HOLD_FOR_BONUS == 4, f"Expected 4 (1 hour on M15), got {MIN_HOLD_FOR_BONUS}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: No bonus on trade open step
# ─────────────────────────────────────────────────────────────────────────────
def test_no_bonus_on_open_step():
    """Opening a position should NOT grant an immediate bonus."""
    env = _make_env()
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    # Open long and capture reward details
    obs, reward, done, trunc, info = env.step(ACTION_OPEN_LONG)
    assert env.position_type == POSITION_LONG

    # The open_bonus component should be 0 on the entry step
    reward_details = getattr(env, '_last_reward_components', {})
    if 'open_bonus' in reward_details:
        assert reward_details['open_bonus'] == 0.0, (
            f"open_bonus should be 0 on entry step, got {reward_details['open_bonus']}"
        )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: No bonus before MIN_HOLD_FOR_BONUS
# ─────────────────────────────────────────────────────────────────────────────
def test_no_bonus_before_threshold():
    """Holding for fewer than MIN_HOLD_FOR_BONUS bars should NOT trigger bonus."""
    env = _make_env()
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    assert env.position_type == POSITION_LONG

    # Hold for MIN_HOLD_FOR_BONUS - 2 bars (should get NO bonus on any of these)
    for i in range(MIN_HOLD_FOR_BONUS - 2):
        obs, reward, done, trunc, info = env.step(ACTION_HOLD)
        reward_details = getattr(env, '_last_reward_components', {})
        if 'open_bonus' in reward_details:
            assert reward_details['open_bonus'] == 0.0, (
                f"open_bonus should be 0 at hold_duration={i+2}, "
                f"got {reward_details['open_bonus']}"
            )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Bonus fires exactly at MIN_HOLD_FOR_BONUS
# ─────────────────────────────────────────────────────────────────────────────
def test_bonus_at_threshold():
    """The deferred bonus should fire exactly when hold_duration == MIN_HOLD_FOR_BONUS."""
    env = _make_env()
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    assert env.position_type == POSITION_LONG
    # After open, current_hold_duration = 1

    # Hold until duration reaches MIN_HOLD_FOR_BONUS
    # We need (MIN_HOLD_FOR_BONUS - 1) more hold steps
    # because hold_duration starts at 1 after open, incremented each step
    bonus_found = False
    for i in range(MIN_HOLD_FOR_BONUS + 2):
        obs, reward, done, trunc, info = env.step(ACTION_HOLD)
        reward_details = getattr(env, '_last_reward_components', {})
        if 'open_bonus' in reward_details and reward_details['open_bonus'] > 0:
            bonus_found = True
            break

    assert bonus_found, (
        f"Deferred entry bonus never fired within {MIN_HOLD_FOR_BONUS + 2} hold steps"
    )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Bonus fires only once (not every step)
# ─────────────────────────────────────────────────────────────────────────────
def test_bonus_fires_once():
    """The deferred bonus should fire on exactly 1 step, not repeatedly."""
    env = _make_env()
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)

    bonus_count = 0
    for _ in range(MIN_HOLD_FOR_BONUS + 10):
        obs, reward, done, trunc, info = env.step(ACTION_HOLD)
        if done:
            break
        reward_details = getattr(env, '_last_reward_components', {})
        if 'open_bonus' in reward_details and reward_details['open_bonus'] > 0:
            bonus_count += 1

    assert bonus_count == 1, (
        f"Deferred bonus should fire exactly once, fired {bonus_count} times"
    )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Churning exploit eliminated
# ─────────────────────────────────────────────────────────────────────────────
def test_churning_exploit_eliminated():
    """Rapid open→close should NOT earn a positive open_bonus.

    OLD behavior: open_bonus=0.3 on every entry → net +0.24/trade.
    NEW behavior: bonus deferred, so immediate close gets 0 bonus.
    """
    env = _make_env()
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    total_open_bonus = 0.0

    # Do 5 rapid open→close cycles (simulating churning)
    for _ in range(5):
        # Open
        obs, r, done, trunc, info = env.step(ACTION_OPEN_LONG)
        if done:
            break
        rd = getattr(env, '_last_reward_components', {})
        total_open_bonus += rd.get('open_bonus', 0.0)

        # Skip cooldown
        _step_n(env, ACTION_HOLD, 3)

        # Close immediately
        obs, r, done, trunc, info = env.step(ACTION_CLOSE_LONG)
        if done:
            break

        # Skip cooldown before next cycle
        _step_n(env, ACTION_HOLD, 3)

    assert total_open_bonus == 0.0, (
        f"Churning exploit still exists! total_open_bonus={total_open_bonus} "
        f"(should be 0 for immediate open→close)"
    )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Short positions also get deferred bonus
# ─────────────────────────────────────────────────────────────────────────────
def test_short_deferred_bonus():
    """The deferred bonus works for short positions too."""
    env = _make_env()
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_SHORT)
    assert env.position_type == POSITION_SHORT

    bonus_found = False
    for i in range(MIN_HOLD_FOR_BONUS + 2):
        obs, reward, done, trunc, info = env.step(ACTION_HOLD)
        reward_details = getattr(env, '_last_reward_components', {})
        if 'open_bonus' in reward_details and reward_details['open_bonus'] > 0:
            bonus_found = True
            break

    assert bonus_found, "Short position should also receive deferred entry bonus"
    env.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
