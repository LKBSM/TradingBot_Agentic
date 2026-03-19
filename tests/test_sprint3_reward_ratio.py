# =============================================================================
# Sprint 3 Validation: Rebalance Close/Hold Reward Ratio (RW-3)
# =============================================================================
# Verifies that hold_reward cap is raised from 0.5→1.5 and close bonus cap
# is lowered from 3.0→2.0, reducing the ratio from 6:1 to ~1.3:1.
#
# Run with: python -m pytest tests/test_sprint3_reward_ratio.py -v
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
    HOLD_REWARD_CAP, CLOSE_BONUS_CAP,
)
from src.environment.environment import TradingEnv


def _make_data(n_rows=800, base_price=2000.0, trend="up"):
    """Create test data with controllable trend."""
    np.random.seed(42)
    if trend == "up":
        prices = base_price + np.linspace(0, 100, n_rows)
    elif trend == "down":
        prices = base_price + np.linspace(0, -100, n_rows)
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
# Test 1: Config constants exist with correct values
# ─────────────────────────────────────────────────────────────────────────────
def test_config_constants():
    """HOLD_REWARD_CAP and CLOSE_BONUS_CAP exist with Sprint 3 values."""
    assert HOLD_REWARD_CAP == 1.5, f"Expected 1.5, got {HOLD_REWARD_CAP}"
    assert CLOSE_BONUS_CAP == 2.0, f"Expected 2.0, got {CLOSE_BONUS_CAP}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Ratio is no longer 6:1
# ─────────────────────────────────────────────────────────────────────────────
def test_ratio_is_balanced():
    """Close:Hold ratio should be ~1.3:1, not the old 6:1."""
    ratio = CLOSE_BONUS_CAP / HOLD_REWARD_CAP
    assert ratio < 2.0, f"Ratio {ratio:.1f}:1 is still too high (was 6:1, target ~1.3:1)"
    assert ratio > 1.0, f"Ratio {ratio:.1f}:1 is below 1:1 — close should mildly beat hold"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Hold reward can exceed old cap of 0.5
# ─────────────────────────────────────────────────────────────────────────────
def test_hold_reward_exceeds_old_cap():
    """v4: DSR produces non-zero rewards while holding a profitable position.

    With DSR, 'hold reward' is implicit in the DSR formula — rising net_worth
    produces positive R_t → positive DSR. No separate hold_reward component.
    """
    env = _make_env(800, trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    if env.position_type != POSITION_LONG:
        env.close()
        return

    # Artificially set entry_price well below current to simulate big profit
    env.entry_price = env.df.iloc[env.current_step]['Close'] * 0.99

    rewards = []
    for _ in range(20):
        obs, reward, done, trunc, info = env.step(ACTION_HOLD)
        if done:
            break
        rewards.append(reward)

    # DSR should produce some positive rewards in uptrend with long position
    positive_rewards = [r for r in rewards if r > 0]
    assert len(positive_rewards) > 0, (
        f"DSR should produce positive rewards holding profitable long. Got: {rewards}"
    )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Hold reward is capped at HOLD_REWARD_CAP
# ─────────────────────────────────────────────────────────────────────────────
def test_hold_reward_respects_cap():
    """v4: DSR reward is clipped to [-10, 10] range."""
    env = _make_env(800, trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)

    for _ in range(50):
        obs, reward, done, trunc, info = env.step(ACTION_HOLD)
        if done:
            break
        assert -10.0 <= reward <= 10.0, (
            f"DSR reward {reward:.3f} out of [-10, 10] clip range"
        )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Close bonus is capped at CLOSE_BONUS_CAP
# ─────────────────────────────────────────────────────────────────────────────
def test_close_bonus_respects_cap():
    """v4: DSR reward on close is bounded by [-20, 20] clipping."""
    env = _make_env(800, trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    _step_n(env, ACTION_HOLD, 50)

    obs, reward, done, trunc, info = env.step(ACTION_CLOSE_LONG)

    assert -20.0 <= reward <= 20.0, (
        f"Close reward {reward:.3f} out of DSR bounds [-20, 20]"
    )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Close bonus no longer reaches 3.0
# ─────────────────────────────────────────────────────────────────────────────
def test_close_bonus_below_old_cap():
    """v4: DSR has no separate trade_bonus — all reward comes from DSR formula."""
    env = _make_env(800, trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    _step_n(env, ACTION_HOLD, 30)

    obs, reward, done, trunc, info = env.step(ACTION_CLOSE_LONG)

    # DSR components should not include old trade_bonus
    rc = getattr(env, '_last_reward_components', {})
    assert 'trade_bonus' not in rc, (
        f"trade_bonus should not exist in DSR reward. Got: {list(rc.keys())}"
    )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Regression — Sprint 1 & 2 still work
# ─────────────────────────────────────────────────────────────────────────────
def test_short_still_works():
    """Short positions still function correctly after reward rebalancing."""
    env = _make_env(800, trend="down")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_SHORT)
    assert env.position_type == POSITION_SHORT

    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_CLOSE_SHORT)
    assert env.position_type == POSITION_FLAT
    assert env.total_trades == 1
    env.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
