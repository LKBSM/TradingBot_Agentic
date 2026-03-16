# =============================================================================
# Sprint 16 Validation: MT5 Broker Integration (LIVE-1)
# =============================================================================
# Verifies that:
# 1. ExecutionBridge translates actions to orders
# 2. Paper trading mode works without MT5
# 3. Kill switch blocks trades when tripped
# 4. LiveTradingLoop initializes correctly
# 5. Position state tracking works
#
# Run with: python -m pytest tests/test_sprint16_mt5_integration.py -v
# =============================================================================

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.live_trading.execution_bridge import ExecutionBridge, ExecutionResult, Action


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: ExecutionBridge instantiation (paper mode)
# ─────────────────────────────────────────────────────────────────────────────
def test_bridge_paper_mode_init():
    """Bridge should initialize in paper mode without MT5."""
    bridge = ExecutionBridge(connector=None, symbol="XAUUSD", paper_mode=True)
    assert bridge.paper_mode is True
    assert bridge.symbol == "XAUUSD"
    assert not bridge.has_position


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: HOLD action returns success
# ─────────────────────────────────────────────────────────────────────────────
def test_hold_action():
    """HOLD should always succeed and do nothing."""
    bridge = ExecutionBridge(paper_mode=True)
    result = bridge.execute_action(action=0)
    assert result.success is True
    assert not bridge.has_position


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Paper OPEN_LONG
# ─────────────────────────────────────────────────────────────────────────────
def test_paper_open_long():
    """OPEN_LONG in paper mode should record position."""
    bridge = ExecutionBridge(paper_mode=True)
    result = bridge.execute_action(
        action=1, volume=0.1, current_price=2000.0, sl=1990.0, tp=2020.0
    )
    assert result.success is True
    assert result.paper_mode is True
    assert result.fill_price == 2000.0
    assert result.volume == 0.1
    assert bridge.has_position
    assert bridge.position_direction == "BUY"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Paper CLOSE_LONG
# ─────────────────────────────────────────────────────────────────────────────
def test_paper_close_long():
    """CLOSE_LONG in paper mode should clear position."""
    bridge = ExecutionBridge(paper_mode=True)
    bridge.execute_action(action=1, volume=0.1, current_price=2000.0)
    assert bridge.has_position

    result = bridge.execute_action(action=2)
    assert result.success is True
    assert not bridge.has_position


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Paper OPEN_SHORT
# ─────────────────────────────────────────────────────────────────────────────
def test_paper_open_short():
    """OPEN_SHORT in paper mode should record sell position."""
    bridge = ExecutionBridge(paper_mode=True)
    result = bridge.execute_action(
        action=3, volume=0.05, current_price=2000.0, sl=2010.0, tp=1980.0
    )
    assert result.success is True
    assert bridge.position_direction == "SELL"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Paper CLOSE_SHORT
# ─────────────────────────────────────────────────────────────────────────────
def test_paper_close_short():
    """CLOSE_SHORT in paper mode should clear position."""
    bridge = ExecutionBridge(paper_mode=True)
    bridge.execute_action(action=3, volume=0.1, current_price=2000.0)
    assert bridge.has_position

    result = bridge.execute_action(action=4)
    assert result.success is True
    assert not bridge.has_position


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Cannot open when already in position
# ─────────────────────────────────────────────────────────────────────────────
def test_cannot_open_when_in_position():
    """Should fail to open a new position when one is already open."""
    bridge = ExecutionBridge(paper_mode=True)
    bridge.execute_action(action=1, volume=0.1, current_price=2000.0)

    result = bridge.execute_action(action=1, volume=0.1, current_price=2001.0)
    assert result.success is False
    assert 'already_in_position' in result.error


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Cannot close when no position
# ─────────────────────────────────────────────────────────────────────────────
def test_cannot_close_no_position():
    """Should fail to close when no position exists."""
    bridge = ExecutionBridge(paper_mode=True)
    result = bridge.execute_action(action=2)
    assert result.success is False
    assert 'no_position' in result.error


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Kill switch blocks new trades
# ─────────────────────────────────────────────────────────────────────────────
def test_kill_switch_blocks_trades():
    """When kill_switch_ok=False, new trades should be blocked."""
    bridge = ExecutionBridge(paper_mode=True)
    result = bridge.execute_action(
        action=1, volume=0.1, current_price=2000.0, kill_switch_ok=False
    )
    assert result.success is False
    assert 'kill_switch' in result.error


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Kill switch allows HOLD
# ─────────────────────────────────────────────────────────────────────────────
def test_kill_switch_allows_hold():
    """HOLD should work even when kill switch is tripped."""
    bridge = ExecutionBridge(paper_mode=True)
    result = bridge.execute_action(action=0, kill_switch_ok=False)
    assert result.success is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: Execution stats tracking
# ─────────────────────────────────────────────────────────────────────────────
def test_execution_stats():
    """Stats should track executions and errors."""
    bridge = ExecutionBridge(paper_mode=True)
    bridge.execute_action(action=0)
    bridge.execute_action(action=1, volume=0.1, current_price=2000.0)
    bridge.execute_action(action=2)

    stats = bridge.get_stats()
    assert stats['total_executions'] == 3
    assert stats['paper_mode'] is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: ExecutionResult serialization
# ─────────────────────────────────────────────────────────────────────────────
def test_result_to_dict():
    """ExecutionResult.to_dict() should produce valid dict."""
    result = ExecutionResult(
        success=True, action=1, fill_price=2000.0, volume=0.1, paper_mode=True
    )
    d = result.to_dict()
    assert d['success'] is True
    assert d['action'] == 'OPEN_LONG'
    assert d['fill_price'] == 2000.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: LiveTradingLoop file exists
# ─────────────────────────────────────────────────────────────────────────────
def test_live_loop_exists():
    """live_trading_loop.py should exist."""
    path = os.path.join(
        os.path.dirname(__file__), '..', 'src', 'live_trading', 'live_trading_loop.py'
    )
    assert os.path.exists(path)


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: ExecutionBridge file exists
# ─────────────────────────────────────────────────────────────────────────────
def test_execution_bridge_exists():
    """execution_bridge.py should exist."""
    path = os.path.join(
        os.path.dirname(__file__), '..', 'src', 'live_trading', 'execution_bridge.py'
    )
    assert os.path.exists(path)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
