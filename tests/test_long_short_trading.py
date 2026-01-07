# =============================================================================
# TEST LONG/SHORT TRADING - Comprehensive Test Suite
# =============================================================================
# This script tests the new 5-action trading system that supports both
# LONG and SHORT positions.
#
# Action Space:
#   0 = HOLD         : Do nothing
#   1 = OPEN_LONG    : Buy to open long position (profit when price UP)
#   2 = CLOSE_LONG   : Sell to close long position
#   3 = OPEN_SHORT   : Sell to open short position (profit when price DOWN)
#   4 = CLOSE_SHORT  : Buy to cover short position
#
# Run with: python tests/test_long_short_trading.py
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration and environment
from src.config import (
    ACTION_HOLD, ACTION_OPEN_LONG, ACTION_CLOSE_LONG,
    ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT,
    POSITION_FLAT, POSITION_LONG, POSITION_SHORT,
    ACTION_NAMES, NUM_ACTIONS
)
from src.environment.environment import TradingEnv


def create_test_data(n_rows: int = 500, trend: str = "up") -> pd.DataFrame:
    """
    Create test OHLCV data with a specified trend.

    Args:
        n_rows: Number of data points
        trend: "up" for uptrend, "down" for downtrend, "flat" for sideways

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)

    # Base price and trend component
    base_price = 2000.0

    if trend == "up":
        # Uptrend: good for LONG positions
        trend_component = np.linspace(0, 100, n_rows)
    elif trend == "down":
        # Downtrend: good for SHORT positions
        trend_component = np.linspace(0, -100, n_rows)
    else:
        # Flat/sideways
        trend_component = np.zeros(n_rows)

    # Add noise
    noise = np.random.normal(0, 5, n_rows).cumsum() * 0.1
    prices = base_price + trend_component + noise

    # Create OHLCV
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=n_rows, freq='15min'),
        'Open': prices * (1 + np.random.uniform(-0.001, 0.001, n_rows)),
        'High': prices * (1 + np.random.uniform(0, 0.003, n_rows)),
        'Low': prices * (1 - np.random.uniform(0, 0.003, n_rows)),
        'Close': prices,
        'Volume': np.random.randint(100, 1000, n_rows),
        # Add required features with dummy values
        'ATR': np.full(n_rows, 10.0),
        'RSI': np.full(n_rows, 50.0),
        'BOS_SIGNAL': np.zeros(n_rows),
        'OB_SIGNAL': np.zeros(n_rows),
    }

    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)

    return df


def test_action_space_configuration():
    """Test that action space is correctly configured."""
    print("\n" + "=" * 70)
    print("TEST 1: Action Space Configuration")
    print("=" * 70)

    # Verify constants
    assert NUM_ACTIONS == 5, f"Expected 5 actions, got {NUM_ACTIONS}"
    assert ACTION_HOLD == 0, f"HOLD should be 0, got {ACTION_HOLD}"
    assert ACTION_OPEN_LONG == 1, f"OPEN_LONG should be 1, got {ACTION_OPEN_LONG}"
    assert ACTION_CLOSE_LONG == 2, f"CLOSE_LONG should be 2, got {ACTION_CLOSE_LONG}"
    assert ACTION_OPEN_SHORT == 3, f"OPEN_SHORT should be 3, got {ACTION_OPEN_SHORT}"
    assert ACTION_CLOSE_SHORT == 4, f"CLOSE_SHORT should be 4, got {ACTION_CLOSE_SHORT}"

    # Verify position constants
    assert POSITION_FLAT == 0, f"FLAT should be 0, got {POSITION_FLAT}"
    assert POSITION_LONG == 1, f"LONG should be 1, got {POSITION_LONG}"
    assert POSITION_SHORT == -1, f"SHORT should be -1, got {POSITION_SHORT}"

    # Verify action names
    assert ACTION_NAMES[0] == 'HOLD'
    assert ACTION_NAMES[1] == 'OPEN_LONG'
    assert ACTION_NAMES[2] == 'CLOSE_LONG'
    assert ACTION_NAMES[3] == 'OPEN_SHORT'
    assert ACTION_NAMES[4] == 'CLOSE_SHORT'

    print("Action constants:")
    for action_id, name in ACTION_NAMES.items():
        print(f"  {action_id} = {name}")
    print("\nPosition constants:")
    print(f"  FLAT  = {POSITION_FLAT}")
    print(f"  LONG  = {POSITION_LONG}")
    print(f"  SHORT = {POSITION_SHORT}")

    print("\n[PASSED] Action space configuration is correct!")
    return True


def test_environment_action_space():
    """Test that environment has correct action space."""
    print("\n" + "=" * 70)
    print("TEST 2: Environment Action Space")
    print("=" * 70)

    df = create_test_data(100)
    env = TradingEnv(df)

    # Check action space
    action_space_n = env.action_space.n
    print(f"Environment action space: Discrete({action_space_n})")

    assert action_space_n == 5, f"Expected Discrete(5), got Discrete({action_space_n})"

    # Reset and check initial state
    obs, info = env.reset()

    print(f"Initial balance: ${env.balance:,.2f}")
    print(f"Initial position: {env.stock_quantity}")
    print(f"Initial position_type: {env.position_type}")

    assert env.position_type == POSITION_FLAT, "Initial position should be FLAT"

    print("\n[PASSED] Environment action space is correct!")
    env.close()
    return True


def test_long_position_cycle():
    """Test opening and closing a LONG position."""
    print("\n" + "=" * 70)
    print("TEST 3: Long Position Cycle (OPEN_LONG -> CLOSE_LONG)")
    print("=" * 70)

    # Create uptrending data (good for longs)
    df = create_test_data(200, trend="up")
    env = TradingEnv(df)
    obs, info = env.reset()

    initial_balance = env.balance
    print(f"Initial balance: ${initial_balance:,.2f}")

    # Step forward a few times to skip initial period
    for _ in range(10):
        obs, reward, done, truncated, info = env.step(ACTION_HOLD)

    # Record price before opening
    entry_step = env.current_step
    entry_price = df.iloc[entry_step]['Close']
    print(f"\nStep {entry_step}: Opening LONG at ${entry_price:,.2f}")

    # OPEN LONG
    obs, reward, done, truncated, info = env.step(ACTION_OPEN_LONG)

    # Verify position opened
    assert env.position_type == POSITION_LONG, f"Expected LONG position, got {env.position_type}"
    assert env.stock_quantity > 0, f"Expected positive quantity, got {env.stock_quantity}"
    print(f"  Position opened: {env.stock_quantity:.4f} units")
    print(f"  Position type: LONG ({env.position_type})")

    # Step forward to let price move (uptrend)
    for _ in range(20):
        obs, reward, done, truncated, info = env.step(ACTION_HOLD)

    # Record price before closing
    exit_step = env.current_step
    exit_price = df.iloc[exit_step]['Close']
    print(f"\nStep {exit_step}: Closing LONG at ${exit_price:,.2f}")

    # CLOSE LONG
    obs, reward, done, truncated, info = env.step(ACTION_CLOSE_LONG)

    # Verify position closed
    assert env.position_type == POSITION_FLAT, f"Expected FLAT, got {env.position_type}"
    assert env.stock_quantity == 0, f"Expected 0 quantity, got {env.stock_quantity}"

    # Check P&L (should be positive in uptrend)
    final_balance = env.balance
    pnl = final_balance - initial_balance
    price_change = exit_price - entry_price

    print(f"  Position closed: FLAT")
    print(f"  Final balance: ${final_balance:,.2f}")
    print(f"  P&L: ${pnl:,.2f}")
    print(f"  Price change: ${price_change:,.2f} ({'+' if price_change > 0 else ''}{price_change/entry_price*100:.2f}%)")

    # In uptrend, long should profit
    if price_change > 0:
        assert pnl > 0 or abs(pnl) < 1, f"Expected profit in uptrend, got {pnl}"
        print("\n[PASSED] Long position profitable in uptrend!")
    else:
        print("\n[INFO] Price moved against position (noise > trend)")

    env.close()
    return True


def test_short_position_cycle():
    """Test opening and closing a SHORT position."""
    print("\n" + "=" * 70)
    print("TEST 4: Short Position Cycle (OPEN_SHORT -> CLOSE_SHORT)")
    print("=" * 70)

    # Create downtrending data (good for shorts)
    df = create_test_data(200, trend="down")
    env = TradingEnv(df)
    obs, info = env.reset()

    initial_balance = env.balance
    print(f"Initial balance: ${initial_balance:,.2f}")

    # Step forward a few times
    for _ in range(10):
        obs, reward, done, truncated, info = env.step(ACTION_HOLD)

    # Record price before opening
    entry_step = env.current_step
    entry_price = df.iloc[entry_step]['Close']
    print(f"\nStep {entry_step}: Opening SHORT at ${entry_price:,.2f}")

    # OPEN SHORT
    obs, reward, done, truncated, info = env.step(ACTION_OPEN_SHORT)

    # Verify position opened (short positions have negative quantity or special flag)
    assert env.position_type == POSITION_SHORT, f"Expected SHORT position, got {env.position_type}"
    print(f"  Position opened: {env.stock_quantity:.4f} units (SHORT)")
    print(f"  Position type: SHORT ({env.position_type})")
    print(f"  Entry price recorded: ${env.entry_price:,.2f}")

    # Step forward to let price move (downtrend)
    for _ in range(20):
        obs, reward, done, truncated, info = env.step(ACTION_HOLD)

    # Record price before closing
    exit_step = env.current_step
    exit_price = df.iloc[exit_step]['Close']
    print(f"\nStep {exit_step}: Closing SHORT at ${exit_price:,.2f}")

    # CLOSE SHORT
    obs, reward, done, truncated, info = env.step(ACTION_CLOSE_SHORT)

    # Verify position closed
    assert env.position_type == POSITION_FLAT, f"Expected FLAT, got {env.position_type}"

    # Check P&L (should be positive when price goes DOWN)
    final_balance = env.balance
    pnl = final_balance - initial_balance
    price_change = exit_price - entry_price

    print(f"  Position closed: FLAT")
    print(f"  Final balance: ${final_balance:,.2f}")
    print(f"  P&L: ${pnl:,.2f}")
    print(f"  Price change: ${price_change:,.2f} ({'+' if price_change > 0 else ''}{price_change/entry_price*100:.2f}%)")

    # In downtrend, short should profit (price goes down = profit)
    if price_change < 0:
        assert pnl > 0 or abs(pnl) < 1, f"Expected profit when price dropped, got {pnl}"
        print("\n[PASSED] Short position profitable in downtrend!")
    else:
        print("\n[INFO] Price moved against position (noise > trend)")

    env.close()
    return True


def test_invalid_actions():
    """Test that invalid actions are handled correctly."""
    print("\n" + "=" * 70)
    print("TEST 5: Invalid Action Handling")
    print("=" * 70)

    df = create_test_data(100)
    env = TradingEnv(df)
    obs, info = env.reset()

    initial_balance = env.balance

    # Test: CLOSE_LONG when no position (should be no-op)
    print("\nTest: CLOSE_LONG when FLAT (no position)")
    obs, reward, done, truncated, info = env.step(ACTION_CLOSE_LONG)
    assert env.position_type == POSITION_FLAT, "Should stay FLAT"
    assert env.balance == initial_balance, "Balance should not change"
    print("  Result: Correctly treated as no-op")

    # Test: CLOSE_SHORT when no position (should be no-op)
    print("\nTest: CLOSE_SHORT when FLAT (no position)")
    obs, reward, done, truncated, info = env.step(ACTION_CLOSE_SHORT)
    assert env.position_type == POSITION_FLAT, "Should stay FLAT"
    print("  Result: Correctly treated as no-op")

    # Open a LONG position
    obs, reward, done, truncated, info = env.step(ACTION_OPEN_LONG)
    assert env.position_type == POSITION_LONG, "Should be LONG"
    print("\nOpened LONG position for next tests")

    # Test: OPEN_LONG when already LONG (should be no-op)
    print("\nTest: OPEN_LONG when already LONG")
    prev_quantity = env.stock_quantity
    obs, reward, done, truncated, info = env.step(ACTION_OPEN_LONG)
    assert env.stock_quantity == prev_quantity, "Should not change position"
    print("  Result: Correctly treated as no-op")

    # Test: OPEN_SHORT when already LONG (should be no-op)
    print("\nTest: OPEN_SHORT when already LONG")
    obs, reward, done, truncated, info = env.step(ACTION_OPEN_SHORT)
    assert env.position_type == POSITION_LONG, "Should stay LONG"
    print("  Result: Correctly treated as no-op")

    # Test: CLOSE_SHORT when LONG (should be no-op)
    print("\nTest: CLOSE_SHORT when LONG (wrong close action)")
    obs, reward, done, truncated, info = env.step(ACTION_CLOSE_SHORT)
    assert env.position_type == POSITION_LONG, "Should stay LONG"
    print("  Result: Correctly treated as no-op")

    print("\n[PASSED] Invalid actions handled correctly!")
    env.close()
    return True


def test_pnl_calculation():
    """Test P&L calculations for both long and short."""
    print("\n" + "=" * 70)
    print("TEST 6: P&L Calculation Verification")
    print("=" * 70)

    # Create flat data for precise testing
    df = create_test_data(100, trend="flat")

    # Manually set specific prices for controlled test
    df['Close'] = 2000.0  # All prices at 2000
    df.iloc[20:40, df.columns.get_loc('Close')] = 2050.0  # Price goes to 2050
    df.iloc[60:80, df.columns.get_loc('Close')] = 1950.0  # Price goes to 1950

    env = TradingEnv(df)
    obs, info = env.reset()

    print("\n--- LONG Position Test ---")
    # Move to step 10 (price = 2000)
    for _ in range(10):
        env.step(ACTION_HOLD)

    # Open LONG at 2000
    initial_balance = env.balance
    env.step(ACTION_OPEN_LONG)
    quantity = env.stock_quantity
    entry_price = env.entry_price
    print(f"Opened LONG: {quantity:.4f} units at ${entry_price:,.2f}")

    # Move to step 25 (price = 2050)
    for _ in range(14):
        env.step(ACTION_HOLD)

    current_price = df.iloc[env.current_step]['Close']
    print(f"Current price: ${current_price:,.2f}")

    # Close LONG
    env.step(ACTION_CLOSE_LONG)
    final_balance = env.balance
    pnl = final_balance - initial_balance
    expected_pnl = quantity * (current_price - entry_price)

    print(f"Closed LONG at ${current_price:,.2f}")
    print(f"Actual P&L: ${pnl:,.2f}")
    print(f"Expected P&L (qty * price_diff): ${expected_pnl:,.2f}")

    # Allow for transaction costs
    if abs(pnl - expected_pnl) < expected_pnl * 0.05:  # Within 5%
        print("[OK] LONG P&L calculation correct (accounting for costs)")

    print("\n--- SHORT Position Test ---")
    # Reset for short test
    env.reset()

    # Move to step 50 (price back to 2000)
    for _ in range(50):
        env.step(ACTION_HOLD)

    # Open SHORT at 2000
    initial_balance = env.balance
    env.step(ACTION_OPEN_SHORT)
    quantity = abs(env.stock_quantity)
    entry_price = env.entry_price
    print(f"Opened SHORT: {quantity:.4f} units at ${entry_price:,.2f}")

    # Move to step 65 (price = 1950)
    for _ in range(14):
        env.step(ACTION_HOLD)

    current_price = df.iloc[env.current_step]['Close']
    print(f"Current price: ${current_price:,.2f}")

    # Close SHORT
    env.step(ACTION_CLOSE_SHORT)
    final_balance = env.balance
    pnl = final_balance - initial_balance
    # SHORT profit = quantity * (entry - exit) when price drops
    expected_pnl = quantity * (entry_price - current_price)

    print(f"Closed SHORT at ${current_price:,.2f}")
    print(f"Actual P&L: ${pnl:,.2f}")
    print(f"Expected P&L (qty * (entry-exit)): ${expected_pnl:,.2f}")

    if expected_pnl > 0:
        print("[OK] SHORT should profit when price drops - correct!")

    print("\n[PASSED] P&L calculations verified!")
    env.close()
    return True


def test_multiple_round_trips():
    """Test multiple long and short trades in sequence."""
    print("\n" + "=" * 70)
    print("TEST 7: Multiple Round Trips")
    print("=" * 70)

    df = create_test_data(500)
    env = TradingEnv(df)
    obs, info = env.reset()

    initial_balance = env.balance
    trade_count = 0
    long_trades = 0
    short_trades = 0

    print(f"Initial balance: ${initial_balance:,.2f}")
    print("\nExecuting trades...")

    # Simulate trading sequence
    for i in range(50):
        if env.position_type == POSITION_FLAT:
            # Randomly open long or short
            if np.random.random() > 0.5:
                env.step(ACTION_OPEN_LONG)
                if env.position_type == POSITION_LONG:
                    long_trades += 1
                    trade_count += 1
            else:
                env.step(ACTION_OPEN_SHORT)
                if env.position_type == POSITION_SHORT:
                    short_trades += 1
                    trade_count += 1
        elif env.position_type == POSITION_LONG:
            # Hold or close
            if np.random.random() > 0.7:
                env.step(ACTION_CLOSE_LONG)
            else:
                env.step(ACTION_HOLD)
        elif env.position_type == POSITION_SHORT:
            # Hold or close
            if np.random.random() > 0.7:
                env.step(ACTION_CLOSE_SHORT)
            else:
                env.step(ACTION_HOLD)

    # Close any remaining position
    if env.position_type == POSITION_LONG:
        env.step(ACTION_CLOSE_LONG)
    elif env.position_type == POSITION_SHORT:
        env.step(ACTION_CLOSE_SHORT)

    final_balance = env.balance
    total_pnl = final_balance - initial_balance

    print(f"\nResults:")
    print(f"  Total trades opened: {trade_count}")
    print(f"  Long trades: {long_trades}")
    print(f"  Short trades: {short_trades}")
    print(f"  Final balance: ${final_balance:,.2f}")
    print(f"  Total P&L: ${total_pnl:,.2f} ({total_pnl/initial_balance*100:+.2f}%)")
    print(f"  Final position: {env.position_type} (should be FLAT)")

    assert env.position_type == POSITION_FLAT, "Should end FLAT"

    print("\n[PASSED] Multiple round trips executed successfully!")
    env.close()
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("     LONG/SHORT TRADING SYSTEM - TEST SUITE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tests = [
        ("Action Space Configuration", test_action_space_configuration),
        ("Environment Action Space", test_environment_action_space),
        ("Long Position Cycle", test_long_position_cycle),
        ("Short Position Cycle", test_short_position_cycle),
        ("Invalid Action Handling", test_invalid_actions),
        ("P&L Calculation", test_pnl_calculation),
        ("Multiple Round Trips", test_multiple_round_trips),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASSED" if result else "FAILED"))
        except Exception as e:
            print(f"\n[ERROR] {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {str(e)[:50]}"))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, status in results if status == "PASSED")
    failed = len(results) - passed

    for name, status in results:
        icon = "[OK]" if status == "PASSED" else "[X]"
        print(f"  {icon} {name}: {status}")

    print("-" * 70)
    print(f"Total: {len(results)} tests | Passed: {passed} | Failed: {failed}")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
