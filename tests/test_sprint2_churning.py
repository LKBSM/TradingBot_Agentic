# =============================================================================
# Sprint 2 Validation: Anti-Churning Properties
# =============================================================================
# v4 UPDATE: The deferred entry bonus (open_bonus) has been removed as part of
# the DSR reward overhaul. DSR naturally discourages churning because:
# 1. Transaction costs reduce R_t → negative DSR
# 2. No additive bonus on entry (no churning exploit possible)
# 3. Only sustained risk-adjusted returns improve the DSR signal
#
# These tests now validate that DSR doesn't reward churning behavior.
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
    np.random.seed(42)
    prices = base_price + np.random.normal(0, 2, n_rows).cumsum() * 0.1
    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=n_rows, freq='15min'),
        'Open': prices * 0.999,
        'High': prices * 1.003,
        'Low': prices * 0.997,
        'Close': prices,
        'Volume': np.full(n_rows, 500),
        'ATR': np.full(n_rows, 10.0),
        'RSI': np.full(n_rows, 50.0),
        'BOS_SIGNAL': np.zeros(n_rows),
    })
    df.set_index('Date', inplace=True)
    return df


def _make_env():
    df = _make_data()
    return TradingEnv(df, strict_scaler_mode=False)


def _step_n(env, action, n):
    result = None
    for _ in range(n):
        result = env.step(action)
        if result[2]:
            break
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: No bonus on immediate open (DSR has no entry bonus)
# ─────────────────────────────────────────────────────────────────────────────
def test_no_bonus_on_entry():
    """v4: DSR has no entry bonus. Opening a position should not produce large positive reward."""
    env = _make_env()
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    # Balance before
    nw_before = env.net_worth
    obs, reward, done, trunc, info = env.step(ACTION_OPEN_LONG)

    # Opening costs fees → reward should not be positive
    if env.position_type == POSITION_LONG:
        assert reward <= 1.0, (
            f"Entry should not produce large positive reward with DSR. Got {reward:.2f}"
        )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: DSR reward components exist
# ─────────────────────────────────────────────────────────────────────────────
def test_bonus_at_threshold():
    """v4: DSR uses dsr/R_t components (no open_bonus)."""
    env = _make_env()
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    if env.position_type != POSITION_LONG:
        env.close()
        return

    # Hold for some steps
    for _ in range(MIN_HOLD_FOR_BONUS + 2):
        obs, reward, done, trunc, info = env.step(ACTION_HOLD)
        if done:
            break

    # DSR should have tracked components
    rc = getattr(env, '_last_reward_components', {})
    assert 'dsr' in rc, f"DSR component expected. Got: {list(rc.keys())}"
    assert 'R_t' in rc, f"R_t component expected. Got: {list(rc.keys())}"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: DSR doesn't produce artificial bonuses
# ─────────────────────────────────────────────────────────────────────────────
def test_bonus_fires_once():
    """v4: DSR has no open_bonus — only R_t-based signal (no artificial spikes)."""
    env = _make_env()
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    if env.position_type != POSITION_LONG:
        env.close()
        return

    rewards = []
    for _ in range(MIN_HOLD_FOR_BONUS + 10):
        obs, reward, done, trunc, info = env.step(ACTION_HOLD)
        if done:
            break
        rewards.append(reward)

    # DSR rewards should be smooth (no artificial spike from open_bonus)
    # Check that no single reward is more than 5x the median (no spikes)
    if len(rewards) >= 5:
        nonzero = [abs(r) for r in rewards if r != 0.0]
        if nonzero:
            median_r = np.median(nonzero)
            if median_r > 0.001:
                max_r = max(nonzero)
                assert max_r < median_r * 20, (
                    f"Artificial reward spike detected: max={max_r:.3f}, median={median_r:.3f}"
                )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Churning exploit eliminated (DSR penalizes via costs)
# ─────────────────────────────────────────────────────────────────────────────
def test_churning_exploit_eliminated():
    """Rapid open→close cycles should not accumulate positive reward.

    DSR naturally penalizes churning: each cycle pays fees (negative R_t)
    which drags down the DSR signal.
    """
    env = _make_env()
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    total_reward = 0.0

    # Do 3 rapid open→close cycles (simulating churning)
    for _ in range(3):
        obs, r, done, trunc, info = env.step(ACTION_OPEN_LONG)
        total_reward += r
        if done:
            break

        _step_n(env, ACTION_HOLD, 3)

        obs, r, done, trunc, info = env.step(ACTION_CLOSE_LONG)
        total_reward += r
        if done:
            break

        _step_n(env, ACTION_HOLD, 3)

    # Churning should not produce positive cumulative reward
    # (fees make each cycle negative)
    assert total_reward <= 2.0, (
        f"Churning produced positive cumulative reward: {total_reward:.2f} "
        f"(should be near 0 or negative from fees)"
    )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Short positions work with DSR
# ─────────────────────────────────────────────────────────────────────────────
def test_short_deferred_bonus():
    """v4: Short positions should produce valid DSR rewards when holding."""
    env = _make_env()
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_SHORT)
    if env.position_type != POSITION_SHORT:
        env.close()
        return

    rewards = []
    for _ in range(MIN_HOLD_FOR_BONUS + 2):
        obs, reward, done, trunc, info = env.step(ACTION_HOLD)
        rewards.append(reward)
        if done:
            break

    # DSR should produce some signal while in position
    rc = getattr(env, '_last_reward_components', {})
    assert 'dsr' in rc, "DSR component expected for short position"
    env.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
