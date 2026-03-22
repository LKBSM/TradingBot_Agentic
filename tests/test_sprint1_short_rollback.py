# =============================================================================
# Sprint 1 Validation: Short Position Rollback Symmetry (SHORT-1)
# =============================================================================
# Verifies that _execute_open_short() and _execute_close_short() now route
# through _execute_trade() with snapshot rollback, matching the long-side pattern.
#
# Run with: python -m pytest tests/test_sprint1_short_rollback.py -v
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, PropertyMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    ACTION_HOLD, ACTION_OPEN_LONG, ACTION_CLOSE_LONG,
    ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT,
    POSITION_FLAT, POSITION_LONG, POSITION_SHORT,
)
from src.environment.environment import TradingEnv


def _make_data(n_rows=200, base_price=2000.0, trend="flat"):
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
        'OB_SIGNAL': np.zeros(n_rows),
    })
    df.set_index('Date', inplace=True)
    return df


def _make_env(n_rows=800, base_price=2000.0, trend="flat"):
    """Create environment with test data (strict scaler disabled for tests)."""
    df = _make_data(n_rows, base_price, trend)
    return TradingEnv(df, strict_scaler_mode=False)


def _step_n(env, action, n):
    """Step environment n times with given action."""
    for _ in range(n):
        obs, r, done, trunc, info = env.step(action)
        if done:
            break
    return obs, r, done, trunc, info


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Open short routes through _execute_trade
# ─────────────────────────────────────────────────────────────────────────────
def test_open_short_calls_execute_trade():
    """Verify _execute_open_short delegates to _execute_trade('sell_to_open')."""
    env = _make_env(800, trend="down")
    env.reset()
    _step_n(env, ACTION_HOLD, 5)

    original_execute = env._execute_trade
    calls = []

    def tracking_execute(trade_type, *args, **kwargs):
        calls.append(trade_type)
        return original_execute(trade_type, *args, **kwargs)

    with patch.object(env, '_execute_trade', side_effect=tracking_execute):
        env.step(ACTION_OPEN_SHORT)

    assert 'sell_to_open' in calls, (
        f"Expected _execute_trade('sell_to_open') call, got: {calls}"
    )
    assert env.position_type == POSITION_SHORT
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Close short routes through _execute_trade
# ─────────────────────────────────────────────────────────────────────────────
def test_close_short_calls_execute_trade():
    """Verify _execute_close_short delegates to _execute_trade('buy_to_cover')."""
    env = _make_env(800, trend="down")
    env.reset()
    _step_n(env, ACTION_HOLD, 5)

    # Open short first
    env.step(ACTION_OPEN_SHORT)
    assert env.position_type == POSITION_SHORT

    _step_n(env, ACTION_HOLD, 5)

    original_execute = env._execute_trade
    calls = []

    def tracking_execute(trade_type, *args, **kwargs):
        calls.append(trade_type)
        return original_execute(trade_type, *args, **kwargs)

    with patch.object(env, '_execute_trade', side_effect=tracking_execute):
        env.step(ACTION_CLOSE_SHORT)

    assert 'buy_to_cover' in calls, (
        f"Expected _execute_trade('buy_to_cover') call, got: {calls}"
    )
    assert env.position_type == POSITION_FLAT
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Rollback on sell_to_open internal exception preserves state
# ─────────────────────────────────────────────────────────────────────────────
def test_open_short_rollback_on_internal_exception():
    """_execute_trade's internal try/except rolls back state on sell_to_open crash."""
    env = _make_env(800)
    env.reset()
    _step_n(env, ACTION_HOLD, 5)

    # Snapshot state before
    balance_before = env.balance
    quantity_before = env.stock_quantity
    fees_before = env.total_fees_paid_episode

    # Inject fault: make trade_commission_pct_of_trade raise inside try block
    # by temporarily replacing it with a property that throws
    real_commission = env.trade_commission_pct_of_trade

    class FakeAttr:
        """Raises on multiplication (used inside sell_to_open branch)."""
        def __rmul__(self, other):
            raise RuntimeError("Injected fault: commission calculation crashed")

    env.trade_commission_pct_of_trade = FakeAttr()

    # Call _execute_trade directly — the internal try/except should catch + rollback
    success, _, _, _, _ = env._execute_trade('sell_to_open', 2000.0, 0.5)

    # Restore for cleanup
    env.trade_commission_pct_of_trade = real_commission

    # Trade must have failed
    assert success is False, "Trade should have failed"

    # State must be rolled back
    assert env.balance == balance_before, (
        f"Balance not rolled back: was {balance_before}, now {env.balance}"
    )
    assert env.stock_quantity == quantity_before, (
        f"Quantity not rolled back: was {quantity_before}, now {env.stock_quantity}"
    )
    assert env.total_fees_paid_episode == fees_before, "Fees not rolled back"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Rollback on buy_to_cover internal exception preserves state
# ─────────────────────────────────────────────────────────────────────────────
def test_close_short_rollback_on_internal_exception():
    """_execute_trade's internal try/except rolls back state on buy_to_cover crash."""
    env = _make_env(800, trend="down")
    env.reset()
    _step_n(env, ACTION_HOLD, 5)

    # Open a real short position first
    env.step(ACTION_OPEN_SHORT)
    assert env.position_type == POSITION_SHORT

    _step_n(env, ACTION_HOLD, 5)

    # Snapshot state while in short position
    balance_in_short = env.balance
    quantity_in_short = env.stock_quantity
    entry_in_short = env.entry_price
    fees_in_short = env.total_fees_paid_episode

    # Inject fault inside buy_to_cover branch
    real_commission = env.trade_commission_pct_of_trade

    class FakeAttr:
        def __rmul__(self, other):
            raise RuntimeError("Injected fault: commission calculation crashed")

    env.trade_commission_pct_of_trade = FakeAttr()

    success, _, _, _, _ = env._execute_trade('buy_to_cover', 1950.0, abs(quantity_in_short))

    env.trade_commission_pct_of_trade = real_commission

    assert success is False, "Trade should have failed"

    # State must be rolled back — still in short position
    assert env.stock_quantity == quantity_in_short, (
        f"Quantity not rolled back: was {quantity_in_short}, now {env.stock_quantity}"
    )
    assert env.balance == balance_in_short, (
        f"Balance not rolled back: was {balance_in_short}, now {env.balance}"
    )
    assert env.total_fees_paid_episode == fees_in_short, "Fees not rolled back"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Full short round-trip P&L (regression)
# ─────────────────────────────────────────────────────────────────────────────
def test_short_roundtrip_pnl():
    """Verify short open → close produces correct directional P&L after refactor.

    This is a regression test — we verify that the refactored code computes
    short P&L with the right sign (entry - exit) and that the position
    lifecycle (SHORT → FLAT) works correctly. Uses cost_multiplier=0 to
    isolate P&L direction from transaction fee noise.
    """
    df = _make_data(800, trend="down")
    env = TradingEnv(df, strict_scaler_mode=False, cost_multiplier=0.0)
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    initial_balance = env.balance
    env.step(ACTION_OPEN_SHORT)

    assert env.position_type == POSITION_SHORT
    entry_price = env.entry_price
    quantity = abs(env.stock_quantity)
    assert quantity > 0, "Should have opened a position"

    # Hold for a few bars
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_CLOSE_SHORT)

    assert env.position_type == POSITION_FLAT, "Should be FLAT after close"
    assert env.stock_quantity == 0.0, "Quantity should be zero"
    assert env.total_trades == 1, "Should have recorded 1 completed trade"

    # Verify trade was recorded in history
    assert len(env.trade_history_summary) == 1
    trade_record = env.trade_history_summary[0]
    assert trade_record['type'] == 'close_short'

    # With zero costs and a downtrend, pnl_abs should be positive
    assert trade_record['pnl_abs'] > 0, \
        f"Short on downtrend with zero costs should be profitable, got pnl={trade_record['pnl_abs']:.4f}"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Long/Short symmetry — both use _execute_trade
# ─────────────────────────────────────────────────────────────────────────────
def test_long_short_symmetry():
    """Both long and short flows route through _execute_trade with rollback."""
    env = _make_env(800)
    env.reset()
    _step_n(env, ACTION_HOLD, 5)

    original_execute = env._execute_trade
    all_calls = []

    def tracking_execute(trade_type, *args, **kwargs):
        all_calls.append(trade_type)
        return original_execute(trade_type, *args, **kwargs)

    with patch.object(env, '_execute_trade', side_effect=tracking_execute):
        # Long round-trip
        env.step(ACTION_OPEN_LONG)
        _step_n(env, ACTION_HOLD, 5)
        env.step(ACTION_CLOSE_LONG)
        _step_n(env, ACTION_HOLD, 5)

        # Short round-trip
        env.step(ACTION_OPEN_SHORT)
        _step_n(env, ACTION_HOLD, 5)
        env.step(ACTION_CLOSE_SHORT)

    # All 4 trade types should appear
    assert 'buy' in all_calls, "OPEN_LONG should call _execute_trade('buy')"
    assert 'sell' in all_calls, "CLOSE_LONG should call _execute_trade('sell')"
    assert 'sell_to_open' in all_calls, "OPEN_SHORT should call _execute_trade('sell_to_open')"
    assert 'buy_to_cover' in all_calls, "CLOSE_SHORT should call _execute_trade('buy_to_cover')"

    assert env.position_type == POSITION_FLAT
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: State snapshot includes all critical fields
# ─────────────────────────────────────────────────────────────────────────────
def test_state_snapshot_completeness():
    """Verify snapshot captures all fields that short trades modify."""
    env = _make_env(800)
    env.reset()

    snapshot = env._create_state_snapshot()

    required_keys = {
        'balance', 'stock_quantity', 'position_type', 'entry_price',
        'net_worth', 'total_fees_paid', 'total_fees_paid_episode',
        'winning_trades', 'losing_trades', 'total_trades',
        'current_hold_duration', 'traded_value_step',
        'transaction_cost_incurred_step',
    }

    missing = required_keys - set(snapshot.keys())
    assert not missing, f"Snapshot missing keys: {missing}"
    env.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
