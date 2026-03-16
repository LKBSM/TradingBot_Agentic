# =============================================================================
# EXECUTION BRIDGE — Environment Actions to MT5 Orders (Sprint 16)
# =============================================================================
# Translates RL agent actions (OPEN_LONG, CLOSE_LONG, etc.) into MT5 broker
# orders. Handles order fill confirmation, SL/TP placement, and paper trading.
#
# Usage:
#   bridge = ExecutionBridge(connector=mt5_conn, symbol="XAUUSD")
#   result = bridge.execute_action(action=1, position_size=0.1)
#
# Paper trading mode:
#   bridge = ExecutionBridge(connector=None, symbol="XAUUSD", paper_mode=True)
# =============================================================================

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import IntEnum

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Agent action space — matches config.py constants."""
    HOLD = 0
    OPEN_LONG = 1
    CLOSE_LONG = 2
    OPEN_SHORT = 3
    CLOSE_SHORT = 4


@dataclass
class ExecutionResult:
    """Result of an action execution attempt."""
    success: bool
    action: int
    fill_price: float = 0.0
    volume: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    ticket: int = 0
    error: str = ""
    paper_mode: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'action': Action(self.action).name,
            'fill_price': self.fill_price,
            'volume': self.volume,
            'sl': self.sl,
            'tp': self.tp,
            'ticket': self.ticket,
            'error': self.error,
            'paper_mode': self.paper_mode,
            'latency_ms': round(self.latency_ms, 2),
        }


class ExecutionBridge:
    """
    Translates RL environment actions to MT5 broker orders.

    Supports:
    - Live execution via MT5Connector
    - Paper trading mode (simulated fills at current price)
    - SL/TP placement via risk manager
    - Kill switch integration (blocks trades when tripped)

    Args:
        connector: MT5Connector instance (None for paper mode)
        symbol: Trading symbol (default "XAUUSD")
        paper_mode: If True, simulate execution without broker
        magic_number: MT5 magic number for bot identification
        max_slippage_points: Maximum allowed slippage in points
    """

    def __init__(
        self,
        connector=None,
        symbol: str = "XAUUSD",
        paper_mode: bool = False,
        magic_number: int = 123456,
        max_slippage_points: int = 20,
    ):
        self.connector = connector
        self.symbol = symbol
        self.paper_mode = paper_mode or (connector is None)
        self.magic_number = magic_number
        self.max_slippage_points = max_slippage_points

        # State tracking
        self._current_ticket: Optional[int] = None
        self._current_direction: Optional[str] = None  # "BUY" or "SELL"
        self._current_volume: float = 0.0
        self._execution_count = 0
        self._error_count = 0

        if self.paper_mode:
            logger.info("ExecutionBridge initialized in PAPER TRADING mode")
        else:
            logger.info(f"ExecutionBridge initialized for {symbol} (live)")

    def execute_action(
        self,
        action: int,
        volume: float = 0.0,
        current_price: float = 0.0,
        sl: float = 0.0,
        tp: float = 0.0,
        kill_switch_ok: bool = True,
    ) -> ExecutionResult:
        """
        Execute an agent action.

        Args:
            action: Action integer (0-4)
            volume: Position size in lots (for open actions)
            current_price: Current market price (required for paper mode)
            sl: Stop loss price
            tp: Take profit price
            kill_switch_ok: Whether kill switch allows trading

        Returns:
            ExecutionResult with fill details
        """
        t0 = time.perf_counter()

        if action == Action.HOLD:
            self._execution_count += 1
            return ExecutionResult(success=True, action=action, paper_mode=self.paper_mode)

        # Kill switch check
        if not kill_switch_ok and action in (Action.OPEN_LONG, Action.OPEN_SHORT):
            logger.warning("Kill switch tripped — blocking new trade")
            return ExecutionResult(
                success=False, action=action, error="kill_switch_blocked",
                paper_mode=self.paper_mode
            )

        try:
            if action == Action.OPEN_LONG:
                result = self._open_position("BUY", volume, current_price, sl, tp)
            elif action == Action.CLOSE_LONG:
                result = self._close_position("BUY")
            elif action == Action.OPEN_SHORT:
                result = self._open_position("SELL", volume, current_price, sl, tp)
            elif action == Action.CLOSE_SHORT:
                result = self._close_position("SELL")
            else:
                result = ExecutionResult(
                    success=False, action=action, error=f"unknown_action_{action}",
                    paper_mode=self.paper_mode
                )
        except Exception as e:
            self._error_count += 1
            logger.error(f"Execution error: {e}", exc_info=True)
            result = ExecutionResult(
                success=False, action=action, error=str(e),
                paper_mode=self.paper_mode
            )

        result.latency_ms = (time.perf_counter() - t0) * 1000
        self._execution_count += 1
        return result

    def _open_position(
        self, direction: str, volume: float, price: float, sl: float, tp: float
    ) -> ExecutionResult:
        """Open a new position (live or paper)."""
        if self._current_ticket is not None:
            return ExecutionResult(
                success=False, action=Action.OPEN_LONG if direction == "BUY" else Action.OPEN_SHORT,
                error="already_in_position", paper_mode=self.paper_mode
            )

        if self.paper_mode:
            # Simulate fill at current price
            self._current_ticket = self._execution_count + 1
            self._current_direction = direction
            self._current_volume = volume
            logger.info(
                f"[PAPER] {direction} {volume:.2f} lots @ {price:.2f} "
                f"SL={sl:.2f} TP={tp:.2f}"
            )
            return ExecutionResult(
                success=True,
                action=Action.OPEN_LONG if direction == "BUY" else Action.OPEN_SHORT,
                fill_price=price, volume=volume, sl=sl, tp=tp,
                ticket=self._current_ticket, paper_mode=True,
            )

        # Live execution via MT5
        order_result = self.connector.open_position(
            symbol=self.symbol,
            direction=direction,
            volume=volume,
            sl=sl if sl > 0 else None,
            tp=tp if tp > 0 else None,
            comment=f"RL_Agent_v2",
            deviation=self.max_slippage_points,
        )

        if order_result.success:
            self._current_ticket = order_result.ticket
            self._current_direction = direction
            self._current_volume = order_result.volume
            logger.info(
                f"[LIVE] {direction} {order_result.volume:.2f} lots "
                f"@ {order_result.price:.2f} ticket={order_result.ticket}"
            )

        return ExecutionResult(
            success=order_result.success,
            action=Action.OPEN_LONG if direction == "BUY" else Action.OPEN_SHORT,
            fill_price=order_result.price,
            volume=order_result.volume,
            sl=sl, tp=tp,
            ticket=order_result.ticket,
            error=order_result.retcode_description if not order_result.success else "",
            paper_mode=False,
        )

    def _close_position(self, expected_direction: str) -> ExecutionResult:
        """Close the current position."""
        action = Action.CLOSE_LONG if expected_direction == "BUY" else Action.CLOSE_SHORT

        if self._current_ticket is None:
            return ExecutionResult(
                success=False, action=action, error="no_position_to_close",
                paper_mode=self.paper_mode
            )

        if self.paper_mode:
            ticket = self._current_ticket
            self._current_ticket = None
            self._current_direction = None
            self._current_volume = 0.0
            logger.info(f"[PAPER] Closed position ticket={ticket}")
            return ExecutionResult(
                success=True, action=action, ticket=ticket, paper_mode=True,
            )

        # Live close via MT5
        order_result = self.connector.close_position(
            ticket=self._current_ticket,
            deviation=self.max_slippage_points,
        )

        if order_result.success:
            logger.info(
                f"[LIVE] Closed ticket={self._current_ticket} "
                f"@ {order_result.price:.2f}"
            )
            self._current_ticket = None
            self._current_direction = None
            self._current_volume = 0.0

        return ExecutionResult(
            success=order_result.success,
            action=action,
            fill_price=order_result.price,
            ticket=order_result.ticket if order_result.success else self._current_ticket,
            error=order_result.retcode_description if not order_result.success else "",
            paper_mode=False,
        )

    @property
    def has_position(self) -> bool:
        """Whether a position is currently open."""
        return self._current_ticket is not None

    @property
    def position_direction(self) -> Optional[str]:
        """Current position direction ("BUY" or "SELL"), or None."""
        return self._current_direction

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'total_executions': self._execution_count,
            'errors': self._error_count,
            'has_position': self.has_position,
            'direction': self._current_direction,
            'volume': self._current_volume,
            'paper_mode': self.paper_mode,
        }
