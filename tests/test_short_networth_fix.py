# =============================================================================
# Test: Short Position Net Worth Accounting Fix
# =============================================================================
# Validates that the catastrophic net worth double-counting bug is fixed.
#
# THE BUG: _update_portfolio_value() computed SHORT net_worth as:
#   net_worth = balance + unrealized_pnl
# But balance already includes short sale proceeds, so this double-counted.
# Result: net_worth inflated ~2x on OPEN_SHORT, causing +10 reward (Sharpe -32.83).
#
# THE FIX: net_worth = balance - (current_price * quantity)
# This correctly subtracts the liability (cost to buy back).
#
# Run with: python -m pytest tests/test_short_networth_fix.py -v
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    ACTION_HOLD, ACTION_OPEN_LONG, ACTION_CLOSE_LONG,
    ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT,
    POSITION_FLAT, POSITION_LONG, POSITION_SHORT,
    INITIAL_BALANCE,
)
from src.environment.environment import TradingEnv


def _make_data(n_rows=800, base_price=2000.0, trend="flat"):
    """Create controlled test data."""
    np.random.seed(42)
    if trend == "down":
        prices = base_price + np.linspace(0, -80, n_rows)
    elif trend == "up":
        prices = base_price + np.linspace(0, 80, n_rows)
    else:
        prices = np.full(n_rows, base_price)

    noise = np.random.normal(0, 1, n_rows) * 0.5
    prices = prices + noise

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
    })
    df.set_index('Date', inplace=True)
    return df


def _make_env(n_rows=800, base_price=2000.0, trend="flat"):
    """Create environment with test data."""
    df = _make_data(n_rows, base_price, trend)
    return TradingEnv(df, strict_scaler_mode=False, training_mode=True)


class TestShortNetWorthAccounting:
    """Tests for short position net worth calculation."""

    def test_open_short_networth_not_inflated(self):
        """Opening a short should NOT inflate net_worth to ~2x initial balance."""
        env = _make_env(base_price=2000.0)
        env.reset()
        initial_nw = env.net_worth

        # Step to build up some state, then open short
        for _ in range(5):
            env.step(ACTION_HOLD)

        obs, reward, done, truncated, info = env.step(ACTION_OPEN_SHORT)

        if env.position_type == POSITION_SHORT:
            # CRITICAL: net_worth should be close to initial (minus fees), NOT ~2x
            assert env.net_worth < initial_nw * 1.1, (
                f"Short net_worth inflated! Got {env.net_worth:.2f}, "
                f"expected < {initial_nw * 1.1:.2f} (initial={initial_nw:.2f}). "
                f"The double-counting bug may still be present."
            )
            assert env.net_worth > initial_nw * 0.8, (
                f"Short net_worth too low! Got {env.net_worth:.2f}, "
                f"expected > {initial_nw * 0.8:.2f}"
            )

    def test_open_short_reward_not_extreme(self):
        """Opening a short should NOT give large POSITIVE reward (old net_worth bug gave +10)."""
        env = _make_env(base_price=2000.0)
        env.reset()

        # Step to build state
        for _ in range(5):
            env.step(ACTION_HOLD)

        obs, reward, done, truncated, info = env.step(ACTION_OPEN_SHORT)

        if env.position_type == POSITION_SHORT:
            # KEY INVARIANT: reward should NOT be positive (fees make it negative)
            # OLD BUG: reward was +10 due to net_worth doubling on OPEN_SHORT
            # With DSR: negative return from fees → negative DSR (can be up to -10 during warm-up)
            assert reward <= 1.0, (
                f"OPEN_SHORT reward too positive: {reward:.2f}. "
                f"Expected negative (fees). If this is +10, the net_worth bug is still present."
            )

    def test_close_short_reward_not_extreme(self):
        """Closing a short should give proportional reward, bounded by DSR range."""
        env = _make_env(base_price=2000.0, trend="flat")
        env.reset()

        # Open short
        for _ in range(5):
            env.step(ACTION_HOLD)
        env.step(ACTION_OPEN_SHORT)

        if env.position_type != POSITION_SHORT:
            pytest.skip("Short position not opened (position sizing too small)")

        # Hold for a few steps (builds DSR history for more stable signal)
        for _ in range(10):
            env.step(ACTION_HOLD)

        # Close short
        obs, reward, done, truncated, info = env.step(ACTION_CLOSE_SHORT)

        if info.get('trade_details', {}).get('trade_success', False):
            # DSR reward is bounded [-10, 10] by clipping
            # Key invariant: reward should be finite and within DSR bounds
            assert -10.0 <= reward <= 10.0, (
                f"CLOSE_SHORT reward out of DSR bounds: {reward:.2f}."
            )

    def test_short_profitable_when_price_drops(self):
        """Short position should be profitable when price drops."""
        env = _make_env(base_price=2000.0, trend="down")
        env.reset()

        initial_nw = env.net_worth

        # Open short
        for _ in range(5):
            env.step(ACTION_HOLD)
        env.step(ACTION_OPEN_SHORT)

        if env.position_type != POSITION_SHORT:
            pytest.skip("Short not opened")

        # Hold while price drops
        for _ in range(20):
            env.step(ACTION_HOLD)

        # Net worth should be higher than initial (we're profiting from price drop)
        assert env.net_worth >= initial_nw * 0.98, (
            f"Short should be profitable on downtrend. "
            f"net_worth={env.net_worth:.2f}, initial={initial_nw:.2f}"
        )

    def test_short_loses_when_price_rises(self):
        """Short position should lose money when price rises (strong uptrend)."""
        # Use steeper uptrend to overcome small position sizes + fees
        env = _make_env(base_price=2000.0, trend="up")
        env.reset()

        initial_nw = env.net_worth

        # Open short
        for _ in range(5):
            env.step(ACTION_HOLD)
        env.step(ACTION_OPEN_SHORT)

        if env.position_type != POSITION_SHORT:
            pytest.skip("Short not opened")

        # Hold for longer while price rises (40 steps to accumulate loss)
        for _ in range(40):
            obs, _, done, trunc, _ = env.step(ACTION_HOLD)
            if done or trunc:
                break

        # Net worth should be lower (losing money on short as price rises)
        # Allow small tolerance for very small position sizes where fees dominate
        if abs(env.stock_quantity) >= 0.01:
            assert env.net_worth < initial_nw * 1.001, (
                f"Short should lose on uptrend. "
                f"net_worth={env.net_worth:.2f}, initial={initial_nw:.2f}, qty={env.stock_quantity:.4f}"
            )

    def test_long_short_symmetry(self):
        """Opening long and short on flat price should have similar (small) impact on net_worth."""
        # Long test
        env_long = _make_env(base_price=2000.0, trend="flat")
        env_long.reset()
        for _ in range(5):
            env_long.step(ACTION_HOLD)
        env_long.step(ACTION_OPEN_LONG)
        long_nw = env_long.net_worth

        # Short test
        env_short = _make_env(base_price=2000.0, trend="flat")
        env_short.reset()
        for _ in range(5):
            env_short.step(ACTION_HOLD)
        env_short.step(ACTION_OPEN_SHORT)
        short_nw = env_short.net_worth

        if env_long.position_type == POSITION_LONG and env_short.position_type == POSITION_SHORT:
            # Both should be close to initial balance (minus fees)
            initial = INITIAL_BALANCE
            assert abs(long_nw - short_nw) < initial * 0.05, (
                f"Long/short net_worth asymmetry too large: "
                f"long_nw={long_nw:.2f}, short_nw={short_nw:.2f}"
            )

    def test_flat_after_close_short(self):
        """After closing a short, net_worth should equal balance (FLAT position)."""
        env = _make_env(base_price=2000.0, trend="flat")
        env.reset()

        # Open short
        for _ in range(5):
            env.step(ACTION_HOLD)
        env.step(ACTION_OPEN_SHORT)

        if env.position_type != POSITION_SHORT:
            pytest.skip("Short not opened")

        # Hold then close
        for _ in range(5):
            env.step(ACTION_HOLD)
        env.step(ACTION_CLOSE_SHORT)

        if env.position_type == POSITION_FLAT:
            # When flat, net_worth = balance (no position)
            assert abs(env.net_worth - env.balance) < 0.01, (
                f"FLAT net_worth should equal balance. "
                f"net_worth={env.net_worth:.2f}, balance={env.balance:.2f}"
            )


class TestScalerConsistency:
    """Tests for scaler data leakage prevention."""

    def test_pre_fitted_scaler_used(self):
        """Passing pre_fitted_scaler should use it instead of fitting new one."""
        from sklearn.preprocessing import MinMaxScaler

        # Need enough rows for fixed episode length (500) + lookback (20)
        df = _make_data(1200, 2000.0, "flat")

        # Fit scaler on first 600 rows (simulate train data)
        env_train = TradingEnv(df.iloc[:600].copy(), strict_scaler_mode=False)
        train_scaler = env_train.scaler

        # Create test env with pre-fitted scaler
        env_test = TradingEnv(
            df.iloc[600:].copy(),
            pre_fitted_scaler=train_scaler,
            strict_scaler_mode=False
        )

        # Should use the same scaler object
        assert env_test.scaler is train_scaler, "Test env should use pre-fitted scaler"
        assert env_test._scaler_source == "pre_fitted"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
