# =============================================================================
# MT5 CONNECTOR - MetaTrader 5 Live Trading Integration
# =============================================================================
# Production-grade connector for executing trades on MetaTrader 5.
#
# Features:
#   - Automatic connection management with reconnection
#   - Order execution (market, limit, stop)
#   - Position management (modify SL/TP, close partial/full)
#   - Account monitoring (balance, equity, margin)
#   - Symbol info and tick data
#   - Trade history retrieval
#   - Magic number filtering for bot isolation
#
# Usage:
#   connector = MT5Connector(
#       account=12345678,
#       password="your_password",
#       server="Broker-Server"
#   )
#   connector.connect()
#   result = connector.open_position("XAUUSD", "BUY", 0.1, sl=2010, tp=2030)
#
# =============================================================================

import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, Thread
import numpy as np

# Try to import MetaTrader5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None


# =============================================================================
# EXCEPTIONS
# =============================================================================

class MT5ConnectionError(Exception):
    """Raised when MT5 connection fails."""
    pass


class MT5OrderError(Exception):
    """Raised when order execution fails."""
    def __init__(self, message: str, retcode: int = None, comment: str = None):
        super().__init__(message)
        self.retcode = retcode
        self.comment = comment


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AccountInfo:
    """MT5 Account Information."""
    login: int
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float  # Equity/Margin as percentage
    leverage: int
    currency: str
    server: str
    trade_allowed: bool
    expert_allowed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            'login': self.login,
            'balance': self.balance,
            'equity': self.equity,
            'margin': self.margin,
            'free_margin': self.free_margin,
            'margin_level': self.margin_level,
            'leverage': self.leverage,
            'currency': self.currency,
            'server': self.server,
            'trade_allowed': self.trade_allowed,
            'expert_allowed': self.expert_allowed,
        }


@dataclass
class PositionInfo:
    """Information about an open position."""
    ticket: int
    symbol: str
    type: str  # "BUY" or "SELL"
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    swap: float
    commission: float
    magic: int
    comment: str
    time_open: datetime

    @property
    def pnl_pips(self) -> float:
        """Calculate P&L in pips."""
        if self.type == "BUY":
            return (self.price_current - self.price_open) * 10  # For XAU
        else:
            return (self.price_open - self.price_current) * 10

    @property
    def duration_hours(self) -> float:
        """Position duration in hours."""
        return (datetime.now() - self.time_open).total_seconds() / 3600

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticket': self.ticket,
            'symbol': self.symbol,
            'type': self.type,
            'volume': self.volume,
            'price_open': self.price_open,
            'price_current': self.price_current,
            'sl': self.sl,
            'tp': self.tp,
            'profit': self.profit,
            'swap': self.swap,
            'commission': self.commission,
            'magic': self.magic,
            'comment': self.comment,
            'time_open': self.time_open.isoformat(),
            'pnl_pips': self.pnl_pips,
            'duration_hours': self.duration_hours,
        }


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    ticket: int = 0
    retcode: int = 0
    retcode_description: str = ""
    volume: float = 0.0
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    comment: str = ""
    request_id: int = 0
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'ticket': self.ticket,
            'retcode': self.retcode,
            'retcode_description': self.retcode_description,
            'volume': self.volume,
            'price': self.price,
            'bid': self.bid,
            'ask': self.ask,
            'comment': self.comment,
            'request_id': self.request_id,
            'execution_time_ms': self.execution_time_ms,
        }


@dataclass
class SymbolInfo:
    """Symbol trading information."""
    name: str
    description: str
    spread: int
    digits: int
    point: float
    trade_contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    trade_stops_level: int  # Minimum SL/TP distance in points
    bid: float
    ask: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'spread': self.spread,
            'digits': self.digits,
            'point': self.point,
            'trade_contract_size': self.trade_contract_size,
            'volume_min': self.volume_min,
            'volume_max': self.volume_max,
            'volume_step': self.volume_step,
            'trade_stops_level': self.trade_stops_level,
            'bid': self.bid,
            'ask': self.ask,
        }


# =============================================================================
# MT5 RETURN CODES (for reference)
# =============================================================================

MT5_RETCODES = {
    10004: "TRADE_RETCODE_REQUOTE - Requote",
    10006: "TRADE_RETCODE_REJECT - Request rejected",
    10007: "TRADE_RETCODE_CANCEL - Request canceled by trader",
    10008: "TRADE_RETCODE_PLACED - Order placed",
    10009: "TRADE_RETCODE_DONE - Request completed",
    10010: "TRADE_RETCODE_DONE_PARTIAL - Only part of the request completed",
    10011: "TRADE_RETCODE_ERROR - Request processing error",
    10012: "TRADE_RETCODE_TIMEOUT - Request timeout",
    10013: "TRADE_RETCODE_INVALID - Invalid request",
    10014: "TRADE_RETCODE_INVALID_VOLUME - Invalid volume",
    10015: "TRADE_RETCODE_INVALID_PRICE - Invalid price",
    10016: "TRADE_RETCODE_INVALID_STOPS - Invalid stops",
    10017: "TRADE_RETCODE_TRADE_DISABLED - Trade disabled",
    10018: "TRADE_RETCODE_MARKET_CLOSED - Market closed",
    10019: "TRADE_RETCODE_NO_MONEY - Not enough money",
    10020: "TRADE_RETCODE_PRICE_CHANGED - Price changed",
    10021: "TRADE_RETCODE_PRICE_OFF - No quotes available",
    10022: "TRADE_RETCODE_INVALID_EXPIRATION - Invalid order expiration",
    10023: "TRADE_RETCODE_ORDER_CHANGED - Order state changed",
    10024: "TRADE_RETCODE_TOO_MANY_REQUESTS - Too many requests",
    10025: "TRADE_RETCODE_NO_CHANGES - No changes in request",
    10026: "TRADE_RETCODE_SERVER_DISABLES_AT - Autotrading disabled by server",
    10027: "TRADE_RETCODE_CLIENT_DISABLES_AT - Autotrading disabled by client",
    10028: "TRADE_RETCODE_LOCKED - Request locked",
    10029: "TRADE_RETCODE_FROZEN - Order frozen",
    10030: "TRADE_RETCODE_INVALID_FILL - Invalid fill type",
    10031: "TRADE_RETCODE_CONNECTION - No connection to server",
    10032: "TRADE_RETCODE_ONLY_REAL - Operation allowed only for live accounts",
    10033: "TRADE_RETCODE_LIMIT_ORDERS - Max pending orders reached",
    10034: "TRADE_RETCODE_LIMIT_VOLUME - Max order volume reached",
    10035: "TRADE_RETCODE_INVALID_ORDER - Invalid order type",
    10036: "TRADE_RETCODE_POSITION_CLOSED - Position already closed",
}


# =============================================================================
# MT5 CONNECTOR
# =============================================================================

class MT5Connector:
    """
    Production-grade MetaTrader 5 connector.

    Provides:
    - Automatic connection management
    - Order execution with retry logic
    - Position management
    - Account monitoring
    - Thread-safe operations

    Example:
        connector = MT5Connector(
            account=12345678,
            password="password",
            server="Broker-Server",
            magic=234001
        )

        if connector.connect():
            # Open a position
            result = connector.open_position(
                symbol="XAUUSD",
                direction="BUY",
                volume=0.1,
                sl=2010.0,
                tp=2030.0,
                comment="Bot signal"
            )

            if result.success:
                print(f"Position opened: ticket={result.ticket}")
    """

    # Default magic number for bot identification
    DEFAULT_MAGIC = 234001

    def __init__(
        self,
        account: int,
        password: str,
        server: str,
        magic: int = None,
        timeout: int = 60000,
        path: str = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize MT5 Connector.

        Args:
            account: MT5 account number
            password: Account password
            server: Broker server name
            magic: Magic number for bot identification
            timeout: Connection timeout in milliseconds
            path: Path to MT5 terminal (optional)
            max_retries: Maximum retry attempts for orders
            retry_delay: Delay between retries in seconds
        """
        if not MT5_AVAILABLE:
            raise ImportError(
                "MetaTrader5 package not installed. "
                "Install with: pip install MetaTrader5"
            )

        self.account = account
        self.password = password
        self.server = server
        self.magic = magic or self.DEFAULT_MAGIC
        self.timeout = timeout
        self.path = path
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._connected = False
        self._lock = Lock()
        self._logger = logging.getLogger("mt5_connector")

        # Statistics
        self._stats = {
            'orders_sent': 0,
            'orders_successful': 0,
            'orders_failed': 0,
            'total_volume': 0.0,
            'reconnections': 0,
        }

    # =========================================================================
    # CONNECTION MANAGEMENT
    # =========================================================================

    def connect(self) -> bool:
        """
        Connect to MT5 terminal and login to account.

        Returns:
            True if connection successful

        Raises:
            MT5ConnectionError if connection fails
        """
        with self._lock:
            # Initialize MT5
            init_kwargs = {"timeout": self.timeout}
            if self.path:
                init_kwargs["path"] = self.path

            if not mt5.initialize(**init_kwargs):
                error = mt5.last_error()
                raise MT5ConnectionError(
                    f"MT5 initialization failed: {error}"
                )

            # Login to account
            if not mt5.login(
                login=self.account,
                password=self.password,
                server=self.server
            ):
                error = mt5.last_error()
                mt5.shutdown()
                raise MT5ConnectionError(
                    f"MT5 login failed: {error}"
                )

            self._connected = True
            self._logger.info(
                f"Connected to MT5: account={self.account}, server={self.server}"
            )

            # Log account info
            account_info = self.get_account_info()
            self._logger.info(
                f"Account: balance=${account_info.balance:.2f}, "
                f"leverage=1:{account_info.leverage}"
            )

            return True

    def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        with self._lock:
            if self._connected:
                mt5.shutdown()
                self._connected = False
                self._logger.info("Disconnected from MT5")

    def reconnect(self) -> bool:
        """Attempt to reconnect to MT5."""
        self._logger.warning("Attempting to reconnect to MT5...")
        self._stats['reconnections'] += 1

        try:
            self.disconnect()
            time.sleep(1)
            return self.connect()
        except Exception as e:
            self._logger.error(f"Reconnection failed: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        if not self._connected:
            return False

        # Verify connection with a simple call
        try:
            info = mt5.terminal_info()
            return info is not None and info.connected
        except Exception:
            self._connected = False
            return False

    def ensure_connected(self) -> bool:
        """Ensure connection is active, reconnect if needed."""
        if not self.is_connected():
            return self.reconnect()
        return True

    # =========================================================================
    # ACCOUNT INFORMATION
    # =========================================================================

    def get_account_info(self) -> AccountInfo:
        """
        Get current account information.

        Returns:
            AccountInfo object with balance, equity, margin, etc.
        """
        if not self.ensure_connected():
            raise MT5ConnectionError("Not connected to MT5")

        info = mt5.account_info()
        if info is None:
            raise MT5ConnectionError("Failed to get account info")

        return AccountInfo(
            login=info.login,
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            margin_level=info.margin_level if info.margin_level else 0.0,
            leverage=info.leverage,
            currency=info.currency,
            server=info.server,
            trade_allowed=info.trade_allowed,
            expert_allowed=info.trade_expert,
        )

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """
        Get symbol trading information.

        Args:
            symbol: Symbol name (e.g., "XAUUSD")

        Returns:
            SymbolInfo object
        """
        if not self.ensure_connected():
            raise MT5ConnectionError("Not connected to MT5")

        # Ensure symbol is available
        if not mt5.symbol_select(symbol, True):
            raise ValueError(f"Symbol {symbol} not available")

        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Failed to get info for {symbol}")

        tick = mt5.symbol_info_tick(symbol)

        return SymbolInfo(
            name=info.name,
            description=info.description,
            spread=info.spread,
            digits=info.digits,
            point=info.point,
            trade_contract_size=info.trade_contract_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            trade_stops_level=info.trade_stops_level,
            bid=tick.bid if tick else 0.0,
            ask=tick.ask if tick else 0.0,
        )

    def get_current_price(self, symbol: str) -> Tuple[float, float]:
        """
        Get current bid/ask prices.

        Returns:
            Tuple of (bid, ask)
        """
        if not self.ensure_connected():
            raise MT5ConnectionError("Not connected to MT5")

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise ValueError(f"No tick data for {symbol}")

        return tick.bid, tick.ask

    # =========================================================================
    # ORDER EXECUTION
    # =========================================================================

    def open_position(
        self,
        symbol: str,
        direction: str,
        volume: float,
        sl: float = None,
        tp: float = None,
        comment: str = None,
        deviation: int = 20
    ) -> OrderResult:
        """
        Open a new position (market order).

        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            direction: "BUY" or "SELL"
            volume: Position size in lots
            sl: Stop loss price (optional)
            tp: Take profit price (optional)
            comment: Order comment (optional)
            deviation: Maximum price deviation in points

        Returns:
            OrderResult with execution details
        """
        if not self.ensure_connected():
            raise MT5ConnectionError("Not connected to MT5")

        start_time = time.time()

        # Validate direction
        direction = direction.upper()
        if direction not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid direction: {direction}")

        # Get current prices
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise MT5OrderError(f"No tick data for {symbol}")

        # Determine order type and price
        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        # Validate volume
        symbol_info = self.get_symbol_info(symbol)
        volume = self._normalize_volume(volume, symbol_info)

        # Validate SL/TP
        if sl is not None:
            sl = self._validate_stop_level(symbol, direction, price, sl, "SL", symbol_info)
        if tp is not None:
            tp = self._validate_stop_level(symbol, direction, price, tp, "TP", symbol_info)

        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "magic": self.magic,
            "comment": comment or f"Bot_{self.magic}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp

        # Execute with retry
        result = self._execute_with_retry(request)

        # Build result
        execution_time = (time.time() - start_time) * 1000

        if result is None:
            return OrderResult(
                success=False,
                retcode=-1,
                retcode_description="Execution failed after retries",
                execution_time_ms=execution_time,
            )

        success = result.retcode == mt5.TRADE_RETCODE_DONE

        # Update statistics
        self._stats['orders_sent'] += 1
        if success:
            self._stats['orders_successful'] += 1
            self._stats['total_volume'] += volume
        else:
            self._stats['orders_failed'] += 1

        return OrderResult(
            success=success,
            ticket=result.order if success else 0,
            retcode=result.retcode,
            retcode_description=MT5_RETCODES.get(result.retcode, f"Unknown: {result.retcode}"),
            volume=result.volume,
            price=result.price,
            bid=result.bid,
            ask=result.ask,
            comment=result.comment,
            request_id=result.request_id,
            execution_time_ms=execution_time,
        )

    def close_position(
        self,
        ticket: int = None,
        symbol: str = None,
        volume: float = None,
        deviation: int = 20,
        comment: str = None
    ) -> OrderResult:
        """
        Close an open position.

        Args:
            ticket: Position ticket to close (if None, closes by symbol)
            symbol: Symbol to close (if ticket not provided)
            volume: Volume to close (None = full position)
            deviation: Maximum price deviation
            comment: Order comment

        Returns:
            OrderResult with execution details
        """
        if not self.ensure_connected():
            raise MT5ConnectionError("Not connected to MT5")

        # Find position
        if ticket is not None:
            positions = mt5.positions_get(ticket=ticket)
        elif symbol is not None:
            positions = mt5.positions_get(symbol=symbol)
        else:
            raise ValueError("Either ticket or symbol must be provided")

        if not positions:
            return OrderResult(
                success=False,
                retcode=-1,
                retcode_description="Position not found",
            )

        # Filter by magic number
        position = None
        for p in positions:
            if p.magic == self.magic:
                position = p
                break

        if position is None:
            # If no bot position found, use first position
            position = positions[0]

        # Determine close direction
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(position.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).ask

        # Volume to close
        close_volume = volume if volume is not None else position.volume
        close_volume = min(close_volume, position.volume)

        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": close_volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": deviation,
            "magic": self.magic,
            "comment": comment or "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        start_time = time.time()
        result = self._execute_with_retry(request)
        execution_time = (time.time() - start_time) * 1000

        if result is None:
            return OrderResult(
                success=False,
                retcode=-1,
                retcode_description="Close failed after retries",
                execution_time_ms=execution_time,
            )

        success = result.retcode == mt5.TRADE_RETCODE_DONE

        return OrderResult(
            success=success,
            ticket=result.order if success else 0,
            retcode=result.retcode,
            retcode_description=MT5_RETCODES.get(result.retcode, f"Unknown: {result.retcode}"),
            volume=result.volume,
            price=result.price,
            execution_time_ms=execution_time,
        )

    def modify_position(
        self,
        ticket: int,
        sl: float = None,
        tp: float = None
    ) -> OrderResult:
        """
        Modify SL/TP of an existing position.

        Args:
            ticket: Position ticket
            sl: New stop loss (None = keep current)
            tp: New take profit (None = keep current)

        Returns:
            OrderResult with execution details
        """
        if not self.ensure_connected():
            raise MT5ConnectionError("Not connected to MT5")

        # Get position
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(
                success=False,
                retcode=-1,
                retcode_description="Position not found",
            )

        position = positions[0]

        # Build request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": sl if sl is not None else position.sl,
            "tp": tp if tp is not None else position.tp,
        }

        start_time = time.time()
        result = mt5.order_send(request)
        execution_time = (time.time() - start_time) * 1000

        if result is None:
            return OrderResult(
                success=False,
                retcode=-1,
                retcode_description="Modify failed",
                execution_time_ms=execution_time,
            )

        success = result.retcode == mt5.TRADE_RETCODE_DONE

        return OrderResult(
            success=success,
            ticket=ticket,
            retcode=result.retcode,
            retcode_description=MT5_RETCODES.get(result.retcode, f"Unknown: {result.retcode}"),
            execution_time_ms=execution_time,
        )

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    def get_positions(
        self,
        symbol: str = None,
        magic_only: bool = True
    ) -> List[PositionInfo]:
        """
        Get open positions.

        Args:
            symbol: Filter by symbol (None = all symbols)
            magic_only: Only return positions opened by this bot

        Returns:
            List of PositionInfo objects
        """
        if not self.ensure_connected():
            raise MT5ConnectionError("Not connected to MT5")

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        result = []
        for p in positions:
            # Filter by magic if requested
            if magic_only and p.magic != self.magic:
                continue

            position_type = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"

            result.append(PositionInfo(
                ticket=p.ticket,
                symbol=p.symbol,
                type=position_type,
                volume=p.volume,
                price_open=p.price_open,
                price_current=p.price_current,
                sl=p.sl,
                tp=p.tp,
                profit=p.profit,
                swap=p.swap,
                commission=0.0,  # Commission is typically on close
                magic=p.magic,
                comment=p.comment,
                time_open=datetime.fromtimestamp(p.time),
            ))

        return result

    def get_position_by_ticket(self, ticket: int) -> Optional[PositionInfo]:
        """Get a specific position by ticket."""
        positions = self.get_positions(magic_only=False)
        for p in positions:
            if p.ticket == ticket:
                return p
        return None

    def close_all_positions(
        self,
        symbol: str = None,
        magic_only: bool = True
    ) -> List[OrderResult]:
        """
        Close all open positions.

        Args:
            symbol: Only close positions for this symbol
            magic_only: Only close positions opened by this bot

        Returns:
            List of OrderResult for each close attempt
        """
        positions = self.get_positions(symbol=symbol, magic_only=magic_only)
        results = []

        for position in positions:
            result = self.close_position(
                ticket=position.ticket,
                comment="Close all"
            )
            results.append(result)

        return results

    # =========================================================================
    # MARKET DATA
    # =========================================================================

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100
    ) -> Optional[np.ndarray]:
        """
        Get OHLCV data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe ("M1", "M5", "M15", "H1", "H4", "D1")
            count: Number of bars

        Returns:
            Numpy array with OHLCV data or None
        """
        if not self.ensure_connected():
            return None

        # Map timeframe string to MT5 constant
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
        }

        mt5_tf = tf_map.get(timeframe.upper())
        if mt5_tf is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        return rates

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _execute_with_retry(self, request: Dict) -> Any:
        """Execute order with retry logic."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = mt5.order_send(request)

                if result is None:
                    self._logger.warning(f"Order send returned None (attempt {attempt + 1})")
                    time.sleep(self.retry_delay)
                    continue

                # Check for retriable errors
                if result.retcode in [10004, 10020, 10021, 10024]:  # Requote, price changed, etc.
                    self._logger.warning(
                        f"Retriable error {result.retcode} (attempt {attempt + 1})"
                    )
                    time.sleep(self.retry_delay)

                    # Update price for next attempt
                    tick = mt5.symbol_info_tick(request['symbol'])
                    if tick:
                        if request['type'] == mt5.ORDER_TYPE_BUY:
                            request['price'] = tick.ask
                        else:
                            request['price'] = tick.bid
                    continue

                return result

            except Exception as e:
                last_error = e
                self._logger.error(f"Order execution error: {e}")
                time.sleep(self.retry_delay)

        if last_error:
            self._logger.error(f"Order failed after {self.max_retries} attempts: {last_error}")

        return None

    def _normalize_volume(self, volume: float, symbol_info: SymbolInfo) -> float:
        """Normalize volume to valid lot size."""
        # Round to volume step
        step = symbol_info.volume_step
        volume = round(volume / step) * step

        # Clamp to limits
        volume = max(symbol_info.volume_min, min(symbol_info.volume_max, volume))

        return round(volume, 2)

    def _validate_stop_level(
        self,
        symbol: str,
        direction: str,
        price: float,
        stop_price: float,
        stop_type: str,
        symbol_info: SymbolInfo
    ) -> float:
        """Validate and adjust stop level if needed."""
        min_distance = symbol_info.trade_stops_level * symbol_info.point

        if direction == "BUY":
            if stop_type == "SL" and stop_price > price - min_distance:
                stop_price = price - min_distance
            elif stop_type == "TP" and stop_price < price + min_distance:
                stop_price = price + min_distance
        else:  # SELL
            if stop_type == "SL" and stop_price < price + min_distance:
                stop_price = price + min_distance
            elif stop_type == "TP" and stop_price > price - min_distance:
                stop_price = price - min_distance

        # Round to symbol digits
        return round(stop_price, symbol_info.digits)

    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            **self._stats,
            'connected': self.is_connected(),
            'magic': self.magic,
            'account': self.account,
            'server': self.server,
        }

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_mt5_connector(
    account: int,
    password: str,
    server: str,
    magic: int = None,
    auto_connect: bool = True
) -> MT5Connector:
    """
    Factory function to create and optionally connect MT5Connector.

    Args:
        account: MT5 account number
        password: Account password
        server: Broker server name
        magic: Magic number (optional)
        auto_connect: Connect immediately if True

    Returns:
        MT5Connector instance
    """
    connector = MT5Connector(
        account=account,
        password=password,
        server=server,
        magic=magic
    )

    if auto_connect:
        connector.connect()

    return connector
