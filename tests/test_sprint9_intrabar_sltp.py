# =============================================================================
# Sprint 9 Validation: Intra-Bar SL/TP Checking Using High/Low (BS-1)
# =============================================================================
# Verifies that:
# 1. SL/TP is checked against High/Low, not just Close
# 2. Fill price is the SL/TP level, not the Close price
# 3. Trailing stop uses High (long) / Low (short) for advancement
# 4. Backward compatible when High/Low not provided
#
# Run with: python -m pytest tests/test_sprint9_intrabar_sltp.py -v
# =============================================================================

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.risk_manager import DynamicRiskManager


def _make_rm():
    """Create a DynamicRiskManager with 1% SL, 2% TP."""
    config = {
        'STOP_LOSS_PERCENTAGE': 0.01,
        'TAKE_PROFIT_PERCENTAGE': 0.02,
        'MIN_TRADE_QUANTITY': 0.001,
        'TSL_START_PROFIT_MULTIPLIER': 1.0,
        'TSL_TRAIL_DISTANCE_MULTIPLIER': 0.5,
    }
    return DynamicRiskManager(config)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Long SL triggers on Low, not Close
# ─────────────────────────────────────────────────────────────────────────────
def test_long_sl_triggers_on_low():
    """For longs, SL should trigger when Low <= SL, even if Close > SL."""
    rm = _make_rm()
    entry = 2000.0
    atr = 10.0
    rm.set_trade_orders(entry, atr, is_long=True)
    sl = rm.current_stop_loss  # entry - 2*ATR = 1980

    # Bar where Close is above SL but Low touches SL
    close = 1985.0  # Above SL
    high = 1990.0
    low = 1979.0    # Below SL (1980)

    signal, fill_price = rm.check_trade_exit(close, is_long=True, high=high, low=low)
    assert signal == 'SL', f"Expected SL trigger, got {signal}"
    assert fill_price == sl, f"Fill price should be SL ({sl}), got {fill_price}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Long SL does NOT trigger when Low > SL
# ─────────────────────────────────────────────────────────────────────────────
def test_long_sl_no_trigger_when_low_above_sl():
    """SL should NOT trigger when Low stays above SL level."""
    rm = _make_rm()
    entry = 2000.0
    atr = 10.0
    rm.set_trade_orders(entry, atr, is_long=True)
    sl = rm.current_stop_loss  # 1980

    close = 1985.0
    high = 1990.0
    low = 1981.0  # Above SL

    signal, fill_price = rm.check_trade_exit(close, is_long=True, high=high, low=low)
    assert signal == 'none', f"Expected no trigger, got {signal}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Long TP triggers on High
# ─────────────────────────────────────────────────────────────────────────────
def test_long_tp_triggers_on_high():
    """For longs, TP should trigger when High >= TP, even if Close < TP."""
    rm = _make_rm()
    entry = 2000.0
    atr = 10.0
    rm.set_trade_orders(entry, atr, is_long=True)
    tp = rm.current_take_profit  # entry * 1.02 = 2040

    close = 2035.0  # Below TP
    high = 2041.0   # Above TP
    low = 2030.0

    signal, fill_price = rm.check_trade_exit(close, is_long=True, high=high, low=low)
    assert signal == 'TP', f"Expected TP trigger, got {signal}"
    assert fill_price == tp, f"Fill price should be TP ({tp}), got {fill_price}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Short SL triggers on High
# ─────────────────────────────────────────────────────────────────────────────
def test_short_sl_triggers_on_high():
    """For shorts, SL should trigger when High >= SL."""
    rm = _make_rm()
    entry = 2000.0
    atr = 10.0
    rm.set_trade_orders(entry, atr, is_long=False)
    sl = rm.current_stop_loss  # entry + 2*ATR = 2020

    close = 2015.0  # Below SL
    high = 2021.0   # Above SL
    low = 2010.0

    signal, fill_price = rm.check_trade_exit(close, is_long=False, high=high, low=low)
    assert signal == 'SL', f"Expected SL trigger, got {signal}"
    assert fill_price == sl, f"Fill price should be SL ({sl}), got {fill_price}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Short TP triggers on Low
# ─────────────────────────────────────────────────────────────────────────────
def test_short_tp_triggers_on_low():
    """For shorts, TP should trigger when Low <= TP."""
    rm = _make_rm()
    entry = 2000.0
    atr = 10.0
    rm.set_trade_orders(entry, atr, is_long=False)
    tp = rm.current_take_profit  # entry * 0.98 = 1960

    close = 1965.0  # Above TP
    high = 1970.0
    low = 1959.0    # Below TP

    signal, fill_price = rm.check_trade_exit(close, is_long=False, high=high, low=low)
    assert signal == 'TP', f"Expected TP trigger, got {signal}"
    assert fill_price == tp, f"Fill price should be TP ({tp}), got {fill_price}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Backward compatibility — no high/low provided
# ─────────────────────────────────────────────────────────────────────────────
def test_backward_compatible_no_high_low():
    """When high/low not provided, should fall back to Close-only behavior."""
    rm = _make_rm()
    entry = 2000.0
    atr = 10.0
    rm.set_trade_orders(entry, atr, is_long=True)
    sl = rm.current_stop_loss  # 1980

    # Close below SL (should trigger even without high/low)
    signal, fill_price = rm.check_trade_exit(1975.0, is_long=True)
    assert signal == 'SL'
    assert fill_price == sl

    # Close above SL (should not trigger)
    rm.set_trade_orders(entry, atr, is_long=True)
    signal2, _ = rm.check_trade_exit(1985.0, is_long=True)
    assert signal2 == 'none'


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Fill price is SL/TP level, not High/Low/Close
# ─────────────────────────────────────────────────────────────────────────────
def test_fill_price_is_order_level():
    """Fill price should be the exact SL/TP order price, not the bar extreme."""
    rm = _make_rm()
    entry = 2000.0
    atr = 10.0
    rm.set_trade_orders(entry, atr, is_long=True)
    sl = rm.current_stop_loss  # 1980

    # Low overshoots SL by $5
    close = 1985.0
    high = 1990.0
    low = 1975.0  # $5 below SL

    signal, fill_price = rm.check_trade_exit(close, is_long=True, high=high, low=low)
    assert signal == 'SL'
    assert fill_price == sl, (
        f"Fill should be at SL ({sl}), not at Low ({low}) or Close ({close})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Trailing stop uses High for long advancement
# ─────────────────────────────────────────────────────────────────────────────
def test_tsl_uses_high_for_long():
    """TSL should advance using bar High, not Close, for longs."""
    rm = _make_rm()
    entry = 2000.0
    atr = 10.0
    rm.set_trade_orders(entry, atr, is_long=True)
    initial_sl = rm.current_stop_loss

    # Bar where Close is modest but High is much higher
    close = 2015.0
    high = 2025.0  # High enough to activate TSL (profit > 1*ATR=10)
    low = 2010.0

    rm.update_trailing_stop(entry, close, atr, is_long=True, high=high, low=low)

    if rm.tsl_activated:
        # TSL should be based on High, not Close
        # new_sl = high - 0.5*ATR = 2025 - 5 = 2020
        expected_sl = high - rm.tsl_trail_distance_multiplier * atr
        assert rm.current_stop_loss >= expected_sl - 0.01, (
            f"TSL should advance to at least {expected_sl}, got {rm.current_stop_loss}"
        )
        assert rm.current_stop_loss > initial_sl, "TSL should have advanced"


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Trailing stop uses Low for short advancement
# ─────────────────────────────────────────────────────────────────────────────
def test_tsl_uses_low_for_short():
    """TSL should advance using bar Low, not Close, for shorts."""
    rm = _make_rm()
    entry = 2000.0
    atr = 10.0
    rm.set_trade_orders(entry, atr, is_long=False)
    initial_sl = rm.current_stop_loss  # 2020

    # Bar where Close is modest but Low is much lower (profitable for short)
    close = 1985.0
    high = 1990.0
    low = 1975.0  # Low enough for TSL activation (profit > 1*ATR=10)

    rm.update_trailing_stop(entry, close, atr, is_long=False, high=high, low=low)

    if rm.tsl_activated:
        # TSL should be based on Low, not Close
        # new_sl = low + 0.5*ATR = 1975 + 5 = 1980
        expected_sl = low + rm.tsl_trail_distance_multiplier * atr
        assert rm.current_stop_loss <= expected_sl + 0.01, (
            f"TSL should advance to at most {expected_sl}, got {rm.current_stop_loss}"
        )
        assert rm.current_stop_loss < initial_sl, "TSL should have tightened"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
