# =============================================================================
# ORDER MANAGER - Trade Lifecycle Management
# =============================================================================
# Manages the complete lifecycle of orders from signal to execution.
#
# Features:
#   - Signal to order conversion
#   - Order validation before execution
#   - Position tracking and reconciliation
#   - SL/TP management and trailing stops
#   - Partial close support
#   - Order history logging
#
# =============================================================================

import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, Thread
import numpy as np

from .mt5_connector import (
    MT5Connector,
    OrderResult,
    PositionInfo,
    AccountInfo
)
from .alerting import AlertManager, AlertLevel


# =============================================================================
# ENUMS
# =============================================================================

class OrderType(Enum):
    """Order types."""
    MARKET_BUY = "BUY"
    MARKET_SELL = "SELL"
    LIMIT_BUY = "LIMIT_BUY"
    LIMIT_SELL = "LIMIT_SELL"
    STOP_BUY = "STOP_BUY"
    STOP_SELL = "STOP_SELL"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    VALIDATING = "VALIDATING"
    EXECUTING = "EXECUTING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class PositionState(Enum):
    """Position state."""
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ManagedOrder:
    """Order with full lifecycle tracking."""
    order_id: str
    symbol: str
    direction: str  # "BUY" or "SELL"
    volume: float
    order_type: OrderType = OrderType.MARKET_BUY

    # Prices
    requested_price: float = 0.0
    executed_price: float = 0.0
    sl: Optional[float] = None
    tp: Optional[float] = None

    # Status
    status: OrderStatus = OrderStatus.PENDING
    ticket: int = 0
    retcode: int = 0
    error_message: str = ""

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    execution_time_ms: float = 0.0

    # Context
    signal_source: str = "bot"
    comment: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'volume': self.volume,
            'order_type': self.order_type.value,
            'requested_price': self.requested_price,
            'executed_price': self.executed_price,
            'sl': self.sl,
            'tp': self.tp,
            'status': self.status.value,
            'ticket': self.ticket,
            'created_at': self.created_at.isoformat(),
            'executed_at': self.executed_at.isoformat() if self.executed_at else None,
            'execution_time_ms': self.execution_time_ms,
        }


@dataclass
class OrderManagerConfig:
    """Configuration for OrderManager."""
    # Validation
    max_slippage_pct: float = 0.1        # 0.1% max slippage
    require_sl: bool = True               # Require stop loss
    require_tp: bool = False              # Require take profit
    min_sl_distance_pct: float = 0.5     # 0.5% minimum SL distance

    # Trailing stop
    enable_trailing_stop: bool = True
    trailing_stop_activation_pct: float = 0.5  # Activate at 0.5% profit
    trailing_stop_distance_pct: float = 0.3    # Trail at 0.3%

    # Risk checks
    max_position_value_pct: float = 20.0  # Max 20% of equity per position
    max_daily_trades: int = 20            # Max trades per day
    cooldown_seconds: int = 60            # Cooldown between trades

    # Execution
    max_execution_attempts: int = 3
    execution_timeout_seconds: int = 30


# =============================================================================
# ORDER MANAGER
# =============================================================================

class OrderManager:
    """
    Manages trade lifecycle from signal to execution.

    Responsibilities:
    1. Validate orders before execution
    2. Execute via MT5Connector
    3. Track open positions
    4. Manage SL/TP and trailing stops
    5. Reconcile positions with broker
    6. Log all order activity

    Usage:
        manager = OrderManager(connector, config)
        manager.start()

        # Execute a trade
        result = manager.execute_signal(
            direction="BUY",
            volume=0.1,
            sl_pct=1.0,  # 1% SL
            tp_pct=2.0,  # 2% TP
        )

        # The manager will:
        # - Validate the order
        # - Calculate exact SL/TP prices
        # - Execute via MT5
        # - Track the position
        # - Manage trailing stop if enabled
    """

    def __init__(
        self,
        connector: MT5Connector,
        config: OrderManagerConfig = None,
        alert_manager: AlertManager = None,
        symbol: str = "XAUUSD"
    ):
        """
        Initialize OrderManager.

        Args:
            connector: MT5Connector instance
            config: OrderManagerConfig
            alert_manager: AlertManager for notifications
            symbol: Default trading symbol
        """
        self.connector = connector
        self.config = config or OrderManagerConfig()
        self.alerts = alert_manager
        self.symbol = symbol

        self._logger = logging.getLogger("OrderManager")
        self._lock = Lock()

        # Order tracking
        self._order_counter = 0
        self._pending_orders: Dict[str, ManagedOrder] = {}
        self._order_history: List[ManagedOrder] = []

        # Position tracking
        self._current_position: Optional[PositionInfo] = None
        self._position_state = PositionState.FLAT

        # Daily tracking
        self._trades_today = 0
        self._last_trade_time: Optional[datetime] = None
        self._daily_reset_date = datetime.now().date()

        # Trailing stop thread
        self._running = False
        self._monitor_thread: Optional[Thread] = None

        # Statistics
        self._stats = {
            'orders_created': 0,
            'orders_executed': 0,
            'orders_rejected': 0,
            'orders_failed': 0,
            'total_volume': 0.0,
            'total_profit': 0.0,
        }

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self):
        """Start the order manager."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self._logger.info("OrderManager started")

    def stop(self):
        """Stop the order manager."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self._logger.info("OrderManager stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                # Sync positions with broker
                self._sync_positions()

                # Update trailing stops
                if self.config.enable_trailing_stop:
                    self._update_trailing_stops()

                # Reset daily counters
                self._check_daily_reset()

                time.sleep(1)  # 1 second interval

            except Exception as e:
                self._logger.error(f"Monitor error: {e}")
                time.sleep(5)

    # =========================================================================
    # ORDER CREATION AND VALIDATION
    # =========================================================================

    def create_order(
        self,
        direction: str,
        volume: float,
        sl: float = None,
        tp: float = None,
        sl_pct: float = None,
        tp_pct: float = None,
        comment: str = None
    ) -> ManagedOrder:
        """
        Create a managed order.

        Args:
            direction: "BUY" or "SELL"
            volume: Position size in lots
            sl: Stop loss price (absolute)
            tp: Take profit price (absolute)
            sl_pct: Stop loss as percentage from entry
            tp_pct: Take profit as percentage from entry
            comment: Order comment

        Returns:
            ManagedOrder ready for execution
        """
        with self._lock:
            self._order_counter += 1
            order_id = f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._order_counter}"

        # Get current price
        bid, ask = self.connector.get_current_price(self.symbol)
        entry_price = ask if direction == "BUY" else bid

        # Calculate SL/TP from percentages if provided
        if sl_pct is not None and sl is None:
            if direction == "BUY":
                sl = entry_price * (1 - sl_pct / 100)
            else:
                sl = entry_price * (1 + sl_pct / 100)

        if tp_pct is not None and tp is None:
            if direction == "BUY":
                tp = entry_price * (1 + tp_pct / 100)
            else:
                tp = entry_price * (1 - tp_pct / 100)

        order = ManagedOrder(
            order_id=order_id,
            symbol=self.symbol,
            direction=direction.upper(),
            volume=volume,
            order_type=OrderType.MARKET_BUY if direction == "BUY" else OrderType.MARKET_SELL,
            requested_price=entry_price,
            sl=sl,
            tp=tp,
            comment=comment or f"Bot_{order_id}",
        )

        self._stats['orders_created'] += 1
        return order

    def validate_order(self, order: ManagedOrder) -> tuple[bool, str]:
        """
        Validate order before execution.

        Returns:
            Tuple of (is_valid, reason)
        """
        order.status = OrderStatus.VALIDATING

        # Check if we're already in a position
        if self._position_state != PositionState.FLAT:
            if order.direction == "BUY" and self._position_state == PositionState.LONG:
                return False, "Already in LONG position"
            if order.direction == "SELL" and self._position_state == PositionState.SHORT:
                return False, "Already in SHORT position"

        # Check daily trade limit
        if self._trades_today >= self.config.max_daily_trades:
            return False, f"Daily trade limit reached ({self.config.max_daily_trades})"

        # Check cooldown
        if self._last_trade_time:
            elapsed = (datetime.now() - self._last_trade_time).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                return False, f"Cooldown active ({self.config.cooldown_seconds - elapsed:.0f}s remaining)"

        # Check SL requirement
        if self.config.require_sl and order.sl is None:
            return False, "Stop loss is required"

        # Check TP requirement
        if self.config.require_tp and order.tp is None:
            return False, "Take profit is required"

        # Check minimum SL distance
        if order.sl is not None:
            sl_distance_pct = abs(order.requested_price - order.sl) / order.requested_price * 100
            if sl_distance_pct < self.config.min_sl_distance_pct:
                return False, f"SL distance ({sl_distance_pct:.2f}%) below minimum ({self.config.min_sl_distance_pct}%)"

        # Check position value limit
        account = self.connector.get_account_info()
        position_value = order.volume * order.requested_price
        position_pct = (position_value / account.equity) * 100

        if position_pct > self.config.max_position_value_pct:
            return False, f"Position value ({position_pct:.1f}%) exceeds limit ({self.config.max_position_value_pct}%)"

        return True, "Order validated"

    # =========================================================================
    # ORDER EXECUTION
    # =========================================================================

    def execute_order(self, order: ManagedOrder) -> OrderResult:
        """
        Execute a validated order.

        Args:
            order: ManagedOrder to execute

        Returns:
            OrderResult from MT5
        """
        order.status = OrderStatus.EXECUTING

        # Execute via connector
        result = self.connector.open_position(
            symbol=order.symbol,
            direction=order.direction,
            volume=order.volume,
            sl=order.sl,
            tp=order.tp,
            comment=order.comment,
        )

        # Update order
        order.executed_at = datetime.now()
        order.execution_time_ms = result.execution_time_ms
        order.executed_price = result.price
        order.ticket = result.ticket
        order.retcode = result.retcode

        if result.success:
            order.status = OrderStatus.FILLED
            self._stats['orders_executed'] += 1
            self._stats['total_volume'] += order.volume

            # Update position state
            self._position_state = (
                PositionState.LONG if order.direction == "BUY"
                else PositionState.SHORT
            )

            # Update counters
            self._trades_today += 1
            self._last_trade_time = datetime.now()

            # Send alert
            if self.alerts:
                self.alerts.trade_opened(
                    symbol=order.symbol,
                    direction=order.direction,
                    volume=order.volume,
                    price=order.executed_price,
                    sl=order.sl,
                    tp=order.tp,
                    ticket=order.ticket,
                )

            self._logger.info(
                f"Order executed: {order.direction} {order.volume} {order.symbol} "
                f"@ {order.executed_price} (ticket={order.ticket})"
            )
        else:
            order.status = OrderStatus.ERROR
            order.error_message = result.retcode_description
            self._stats['orders_failed'] += 1

            self._logger.error(
                f"Order failed: {order.order_id} - {result.retcode_description}"
            )

        # Archive order
        self._order_history.append(order)

        return result

    def execute_signal(
        self,
        direction: str,
        volume: float,
        sl_pct: float = 1.0,
        tp_pct: float = 2.0,
        comment: str = None
    ) -> tuple[bool, OrderResult | str]:
        """
        Execute a trading signal (convenience method).

        Args:
            direction: "BUY" or "SELL"
            volume: Position size in lots
            sl_pct: Stop loss percentage
            tp_pct: Take profit percentage
            comment: Order comment

        Returns:
            Tuple of (success, result_or_error_message)
        """
        # Create order
        order = self.create_order(
            direction=direction,
            volume=volume,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            comment=comment,
        )

        # Validate
        is_valid, reason = self.validate_order(order)
        if not is_valid:
            order.status = OrderStatus.REJECTED
            order.error_message = reason
            self._stats['orders_rejected'] += 1
            self._order_history.append(order)
            return False, reason

        # Execute
        result = self.execute_order(order)
        return result.success, result

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    def close_position(
        self,
        volume: float = None,
        comment: str = None
    ) -> tuple[bool, OrderResult | str]:
        """
        Close current position.

        Args:
            volume: Volume to close (None = full)
            comment: Order comment

        Returns:
            Tuple of (success, result_or_error)
        """
        if self._position_state == PositionState.FLAT:
            return False, "No position to close"

        positions = self.connector.get_positions(symbol=self.symbol)
        if not positions:
            self._position_state = PositionState.FLAT
            return False, "Position not found on broker"

        position = positions[0]

        result = self.connector.close_position(
            ticket=position.ticket,
            volume=volume,
            comment=comment or "Close position",
        )

        if result.success:
            # Calculate profit
            profit = position.profit

            # Update state
            if volume is None or volume >= position.volume:
                self._position_state = PositionState.FLAT

            self._stats['total_profit'] += profit

            # Alert
            if self.alerts:
                self.alerts.trade_closed(
                    symbol=position.symbol,
                    direction=position.type,
                    volume=volume or position.volume,
                    profit=profit,
                    ticket=position.ticket,
                )

            self._logger.info(f"Position closed: profit=${profit:.2f}")

        return result.success, result

    def _sync_positions(self):
        """Synchronize position state with broker."""
        try:
            positions = self.connector.get_positions(symbol=self.symbol)

            if not positions:
                if self._position_state != PositionState.FLAT:
                    self._logger.info("Position closed externally")
                self._position_state = PositionState.FLAT
                self._current_position = None
            else:
                self._current_position = positions[0]
                self._position_state = (
                    PositionState.LONG if positions[0].type == "BUY"
                    else PositionState.SHORT
                )

        except Exception as e:
            self._logger.error(f"Position sync error: {e}")

    # =========================================================================
    # TRAILING STOP
    # =========================================================================

    def _update_trailing_stops(self):
        """Update trailing stop for open positions."""
        if self._current_position is None:
            return

        position = self._current_position
        entry_price = position.price_open
        current_price = position.price_current

        # Calculate profit percentage
        if position.type == "BUY":
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100

        # Check if trailing should activate
        if profit_pct < self.config.trailing_stop_activation_pct:
            return

        # Calculate new SL
        trail_distance = current_price * (self.config.trailing_stop_distance_pct / 100)

        if position.type == "BUY":
            new_sl = current_price - trail_distance
            # Only move SL up
            if position.sl is None or new_sl > position.sl:
                self._modify_sl(position.ticket, new_sl)
        else:
            new_sl = current_price + trail_distance
            # Only move SL down
            if position.sl is None or new_sl < position.sl:
                self._modify_sl(position.ticket, new_sl)

    def _modify_sl(self, ticket: int, new_sl: float):
        """Modify stop loss of a position."""
        try:
            result = self.connector.modify_position(ticket=ticket, sl=new_sl)
            if result.success:
                self._logger.debug(f"Trailing SL updated: ticket={ticket}, sl={new_sl:.2f}")
        except Exception as e:
            self._logger.error(f"Failed to update trailing SL: {e}")

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _check_daily_reset(self):
        """Reset daily counters at midnight."""
        today = datetime.now().date()
        if today != self._daily_reset_date:
            self._trades_today = 0
            self._daily_reset_date = today
            self._logger.info("Daily counters reset")

    def get_position_state(self) -> PositionState:
        """Get current position state."""
        return self._position_state

    def get_current_position(self) -> Optional[PositionInfo]:
        """Get current position info."""
        return self._current_position

    def get_statistics(self) -> Dict[str, Any]:
        """Get order manager statistics."""
        return {
            **self._stats,
            'position_state': self._position_state.value,
            'trades_today': self._trades_today,
            'orders_in_history': len(self._order_history),
        }

    def get_order_history(self, limit: int = 50) -> List[Dict]:
        """Get recent order history."""
        return [o.to_dict() for o in self._order_history[-limit:]]
