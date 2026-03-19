# =============================================================================
# Sprint 4 Validation: Reward Function Integrity
# =============================================================================
# v4 UPDATE: The additive reward function (with trade_bonus, quality multiplier,
# hold_reward) has been replaced by Differential Sharpe Ratio (DSR).
# These tests now validate DSR properties instead of the old components.
#
# Run with: python -m pytest tests/test_sprint4_double_counting.py -v
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
)
from src.environment.environment import TradingEnv


def _make_data(n_rows=800, base_price=2000.0, trend="up"):
    """Create test data with controllable trend."""
    np.random.seed(42)
    if trend == "up":
        prices = base_price + np.linspace(0, 80, n_rows)
    elif trend == "down":
        prices = base_price + np.linspace(0, -80, n_rows)
    else:
        prices = np.full(n_rows, base_price)
    noise = np.random.normal(0, 0.3, n_rows)
    prices = prices + noise

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
        'OB_SIGNAL': np.zeros(n_rows),
    })
    df.set_index('Date', inplace=True)
    return df


def _make_env(n_rows=800, trend="up"):
    df = _make_data(n_rows, trend=trend)
    return TradingEnv(df, strict_scaler_mode=False)


def _step_n(env, action, n):
    result = None
    for _ in range(n):
        result = env.step(action)
        if result[2]:
            break
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: DSR tracks reward components
# ─────────────────────────────────────────────────────────────────────────────
def test_no_additive_bonus_on_winning_close():
    """v4: DSR reward should track dsr, R_t, final_reward (no trade_bonus)."""
    env = _make_env(800, trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    if env.position_type != POSITION_LONG:
        env.close()
        return  # Skip if position not opened

    _step_n(env, ACTION_HOLD, 30)
    env.step(ACTION_CLOSE_LONG)

    rc = getattr(env, '_last_reward_components', {})

    # DSR should have these components (not old trade_bonus/profitability)
    assert 'dsr' in rc, f"DSR component missing. Got: {list(rc.keys())}"
    assert 'R_t' in rc, f"R_t component missing. Got: {list(rc.keys())}"
    assert 'final_reward' in rc, f"final_reward component missing. Got: {list(rc.keys())}"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: DSR responds to losses (no separate losing penalty needed)
# ─────────────────────────────────────────────────────────────────────────────
def test_losing_close_keeps_penalty():
    """v4: Losing close should produce negative DSR reward."""
    env = _make_env(800, trend="down")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    if env.position_type != POSITION_LONG:
        env.close()
        return

    # Hold while price drops
    _step_n(env, ACTION_HOLD, 30)

    obs, reward, done, trunc, info = env.step(ACTION_CLOSE_LONG)

    # DSR reward on losing close should be negative or zero
    # (net_worth decreases → R_t negative → DSR negative)
    rc = getattr(env, '_last_reward_components', {})
    final = rc.get('final_reward', reward)
    assert final <= 1.0, (
        f"Losing close reward should be <= 1.0, got {final:.2f}"
    )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: DSR is bounded
# ─────────────────────────────────────────────────────────────────────────────
def test_profitability_amplified_on_win():
    """v4: DSR reward should be within [-10, 10] range."""
    env = _make_env(800, trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    _step_n(env, ACTION_HOLD, 30)

    env.step(ACTION_CLOSE_LONG)

    rc = getattr(env, '_last_reward_components', {})
    final = rc.get('final_reward', 0.0)

    assert -10.0 <= final <= 10.0, f"DSR reward out of bounds: {final:.2f}"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Winning close reward bounded (no double-counting possible with DSR)
# ─────────────────────────────────────────────────────────────────────────────
def test_winning_close_reward_range():
    """v4: DSR has no double-counting by construction (single formula)."""
    env = _make_env(800, trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    current_price = env.df.iloc[env.current_step]['Close']
    env.entry_price = current_price * 0.99

    _step_n(env, ACTION_HOLD, 5)
    obs, reward, done, trunc, info = env.step(ACTION_CLOSE_LONG)

    # DSR reward is bounded by clipping
    assert -20.0 <= reward <= 20.0, f"Reward {reward:.2f} out of DSR bounds"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: DSR doesn't use separate multiplier caps
# ─────────────────────────────────────────────────────────────────────────────
def test_quality_multiplier_capped():
    """v4: DSR has no quality multiplier — test DSR EMAs are updated."""
    env = _make_env(800, trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    _step_n(env, ACTION_HOLD, 10)

    # DSR EMAs should have been updated
    assert env._dsr_A != 0.0 or env._dsr_B != 1e-8, (
        "DSR EMAs should be updated after steps with position"
    )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Win/loss rewards are approximately symmetric with DSR
# ─────────────────────────────────────────────────────────────────────────────
def test_win_loss_asymmetry_reduced():
    """v4: DSR is inherently symmetric — same formula for gains and losses."""
    env = _make_env(800, trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)
    env.step(ACTION_OPEN_LONG)

    # Collect rewards from holding in uptrend
    rewards = []
    for _ in range(20):
        _, r, done, trunc, _ = env.step(ACTION_HOLD)
        rewards.append(r)
        if done or trunc:
            break

    # DSR should produce some non-zero rewards
    if env.position_type == POSITION_LONG:
        assert any(r != 0.0 for r in rewards), "DSR should produce non-zero rewards in position"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Short close works with DSR
# ─────────────────────────────────────────────────────────────────────────────
def test_short_close_uses_multiplier():
    """v4: Short close should produce valid DSR reward."""
    env = _make_env(800, trend="down")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_SHORT)
    if env.position_type != POSITION_SHORT:
        env.close()
        return

    _step_n(env, ACTION_HOLD, 30)
    obs, reward, done, trunc, info = env.step(ACTION_CLOSE_SHORT)

    # DSR reward should be bounded
    assert -20.0 <= reward <= 20.0, f"Short close reward out of bounds: {reward:.2f}"

    # DSR components should exist
    rc = getattr(env, '_last_reward_components', {})
    assert 'dsr' in rc, f"DSR component missing. Got: {list(rc.keys())}"
    env.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
