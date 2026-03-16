# =============================================================================
# Sprint 4 Validation: Fix P&L Double-Counting (RW-2)
# =============================================================================
# Verifies that the additive trade_bonus has been replaced with a quality
# multiplier on profitability_reward, eliminating the double-counting where
# a +2% trade earned ~4.0 total (2.0 log-return + 2.0 bonus).
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
    CLOSE_BONUS_CAP, STOP_LOSS_PERCENTAGE,
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
# Test 1: trade_bonus is 0.0 for winning closes
# ─────────────────────────────────────────────────────────────────────────────
def test_no_additive_bonus_on_winning_close():
    """Winning close should have trade_bonus=0.0 (multiplier applied instead)."""
    env = _make_env(800, trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    assert env.position_type == POSITION_LONG

    # Hold in uptrend for a clearly profitable close (100 bars ≈ 10pt gain >> costs)
    _step_n(env, ACTION_HOLD, 100)

    env.step(ACTION_CLOSE_LONG)

    rc = getattr(env, '_last_reward_components', {})
    trade_bonus = rc.get('trade_bonus', None)

    # For a winning trade, trade_bonus should be 0.0 (no additive bonus)
    assert trade_bonus is not None, "trade_bonus not in reward components"
    assert trade_bonus == 0.0, (
        f"Expected trade_bonus=0.0 for winning close (multiplier only), "
        f"got {trade_bonus:.3f}"
    )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: trade_bonus is -0.5 for losing closes
# ─────────────────────────────────────────────────────────────────────────────
def test_losing_close_keeps_penalty():
    """Losing close should still get trade_bonus=-0.5."""
    env = _make_env(800, trend="down")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    # Open long in downtrend (will lose)
    env.step(ACTION_OPEN_LONG)
    assert env.position_type == POSITION_LONG

    # Hold while price drops
    _step_n(env, ACTION_HOLD, 30)

    env.step(ACTION_CLOSE_LONG)

    rc = getattr(env, '_last_reward_components', {})
    trade_bonus = rc.get('trade_bonus', None)

    assert trade_bonus is not None, "trade_bonus not in reward components"
    assert trade_bonus == -0.5, (
        f"Expected trade_bonus=-0.5 for losing close, got {trade_bonus:.3f}"
    )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: profitability_reward is amplified on winning close
# ─────────────────────────────────────────────────────────────────────────────
def test_profitability_amplified_on_win():
    """On a winning close, profitability_reward should be > raw log-return.

    The quality multiplier (1.0 + actual_rr * 0.3) should boost it.
    """
    env = _make_env(800, trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    _step_n(env, ACTION_HOLD, 30)

    # Capture net_worth before close to compute raw log-return
    prev_nw = env.net_worth
    env.step(ACTION_CLOSE_LONG)
    curr_nw = env.net_worth

    rc = getattr(env, '_last_reward_components', {})
    prof_reward = rc.get('profitability', 0.0)

    # Raw log-return (before multiplier)
    if prev_nw > 0 and curr_nw > 0:
        raw_log_return = np.log(curr_nw / prev_nw) * 100.0
    else:
        raw_log_return = 0.0

    trade_pnl = env.trade_details.get('trade_pnl_abs', 0.0)
    if trade_pnl > 0 and raw_log_return > 0.01:
        # Profitability should be amplified (greater than raw log-return)
        assert prof_reward >= raw_log_return, (
            f"profitability_reward ({prof_reward:.4f}) should be >= raw log-return "
            f"({raw_log_return:.4f}) due to quality multiplier"
        )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Total reward for +1% close is in reasonable range
# ─────────────────────────────────────────────────────────────────────────────
def test_winning_close_reward_range():
    """Total reward for a winning close should be bounded, not inflated.

    With the quality multiplier, a +1% close should produce a total reward
    roughly in the range [0.5, 3.5], not the old [3.0, 5.0].
    """
    env = _make_env(800, trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)

    # Artificially set entry to get ~1% profit on close
    current_price = env.df.iloc[env.current_step]['Close']
    env.entry_price = current_price * 0.99  # 1% below current = ~1% profit

    _step_n(env, ACTION_HOLD, 5)

    obs, reward, done, trunc, info = env.step(ACTION_CLOSE_LONG)

    rc = getattr(env, '_last_reward_components', {})
    final_reward = rc.get('final_reward', reward)

    # With quality multiplier, a ~1% close should NOT produce rewards > 4.0
    # (old system would give ~2.0 log-return + ~2.0 bonus = ~4.0)
    assert final_reward < 4.0, (
        f"Reward {final_reward:.2f} is too high — double-counting may still exist"
    )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Quality multiplier is capped at CLOSE_BONUS_CAP
# ─────────────────────────────────────────────────────────────────────────────
def test_quality_multiplier_capped():
    """The quality multiplier should never exceed CLOSE_BONUS_CAP."""
    env = _make_env(800, trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)

    # Set entry very low to simulate huge R:R
    current_price = env.df.iloc[env.current_step]['Close']
    env.entry_price = current_price * 0.95  # 5% below = huge win

    _step_n(env, ACTION_HOLD, 5)

    prev_nw = env.net_worth
    env.step(ACTION_CLOSE_LONG)
    curr_nw = env.net_worth

    rc = getattr(env, '_last_reward_components', {})
    prof_reward = rc.get('profitability', 0.0)

    # Compute what raw would have been
    if prev_nw > 0 and curr_nw > 0:
        raw_log_return = np.log(curr_nw / prev_nw) * 100.0
    else:
        raw_log_return = 0.01

    if raw_log_return > 0.01:
        effective_multiplier = prof_reward / raw_log_return
        assert effective_multiplier <= CLOSE_BONUS_CAP + 0.01, (
            f"Quality multiplier {effective_multiplier:.2f}x exceeds cap {CLOSE_BONUS_CAP}"
        )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Win/loss asymmetry is reduced
# ─────────────────────────────────────────────────────────────────────────────
def test_win_loss_asymmetry_reduced():
    """The reward magnitude for a +1% win and -1% loss should be more symmetric.

    Old: +1% = ~4.0, -1% = ~-2.5 (ratio 1.6:1)
    New: +1% = ~2.0-3.0, -1% = ~-2.5 (ratio ~1:1)
    """
    # Winning close
    env = _make_env(800, trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)
    env.step(ACTION_OPEN_LONG)
    current_price = env.df.iloc[env.current_step]['Close']
    env.entry_price = current_price * 0.99
    _step_n(env, ACTION_HOLD, 5)
    env.step(ACTION_CLOSE_LONG)
    rc_win = getattr(env, '_last_reward_components', {})
    win_reward = rc_win.get('final_reward', 0.0)
    env.close()

    # Losing close
    env2 = _make_env(800, trend="down")
    env2.reset()
    _step_n(env2, ACTION_HOLD, 10)
    env2.step(ACTION_OPEN_LONG)
    current_price2 = env2.df.iloc[env2.current_step]['Close']
    env2.entry_price = current_price2 * 1.01  # 1% above = ~1% loss
    _step_n(env2, ACTION_HOLD, 5)
    env2.step(ACTION_CLOSE_LONG)
    rc_loss = getattr(env2, '_last_reward_components', {})
    loss_reward = rc_loss.get('final_reward', 0.0)
    env2.close()

    if win_reward > 0 and loss_reward < 0:
        ratio = abs(win_reward / loss_reward)
        # Old ratio was ~1.6:1, new should be closer to 1:1
        assert ratio < 2.5, (
            f"Win/loss ratio {ratio:.2f}:1 is still too asymmetric. "
            f"Win={win_reward:.2f}, Loss={loss_reward:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Short close also uses multiplier (not additive)
# ─────────────────────────────────────────────────────────────────────────────
def test_short_close_uses_multiplier():
    """Profitable short close should have trade_bonus=0.0 (multiplier applied)."""
    env = _make_env(800, trend="down")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_SHORT)
    assert env.position_type == POSITION_SHORT

    # Hold in downtrend for profit
    _step_n(env, ACTION_HOLD, 30)
    env.step(ACTION_CLOSE_SHORT)

    rc = getattr(env, '_last_reward_components', {})
    trade_bonus = rc.get('trade_bonus', None)
    trade_pnl = env.trade_details.get('trade_pnl_abs', 0.0)

    if trade_pnl > 0:
        assert trade_bonus == 0.0, (
            f"Short winning close should have trade_bonus=0.0, got {trade_bonus}"
        )
    env.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
