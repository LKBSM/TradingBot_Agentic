# =============================================================================
# MT5 CONNECTION POOL - Optimized MT5 Connection Management
# =============================================================================
# Reduces connection overhead by reusing MT5 connections.
#
# Features:
#   - Connection pooling with configurable size
#   - Automatic connection health checks
#   - Reconnection on failure
#   - Thread-safe connection checkout
#   - Latency tracking per connection
#   - Graceful degradation
#
# Usage:
#   pool = MT5ConnectionPool(config)
#   pool.start()
#
#   with pool.get_connection() as conn:
#       result = conn.order_send(request)
#
#   pool.stop()
#
# =============================================================================

import os
import time
import logging
import threading
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass, field
from contextlib import contextmanager
from datetime import datetime, timedelta
from queue import Queue, Empty
from enum import Enum

# Optional MT5 import
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None


# =============================================================================
# CONFIGURATION
# =============================================================================

class ConnectionState(Enum):
    """Connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class MT5PoolConfig:
    """Configuration for MT5 connection pool."""
    # Connection settings
    pool_size: int = 4
    login: int = 0
    password: str = ""
    server: str = ""
    path: str = ""  # Path to terminal64.exe

    # Timeout settings
    connection_timeout_seconds: float = 30.0
    checkout_timeout_seconds: float = 5.0
    operation_timeout_seconds: float = 10.0

    # Health check settings
    health_check_interval_seconds: float = 30.0
    max_consecutive_failures: int = 3

    # Retry settings
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0

    @classmethod
    def from_environment(cls) -> 'MT5PoolConfig':
        """Create config from environment variables."""
        return cls(
            pool_size=int(os.getenv('MT5_POOL_SIZE', '4')),
            login=int(os.getenv('MT5_LOGIN', '0')),
            password=os.getenv('MT5_PASSWORD', ''),
            server=os.getenv('MT5_SERVER', ''),
            path=os.getenv('MT5_PATH', ''),
        )

    @classmethod
    def from_secret_manager(cls, secret_manager, pool_size: int = 4) -> 'MT5PoolConfig':
        """Create config from Sprint 1 SecretManager."""
        creds = secret_manager.get_mt5_credentials()
        return cls(
            pool_size=pool_size,
            login=creds['account'],
            password=creds['password'],
            server=creds['server'],
        )


# =============================================================================
# POOLED CONNECTION
# =============================================================================

@dataclass
class PooledConnection:
    """A pooled MT5 connection with metadata."""
    id: int
    state: ConnectionState = ConnectionState.DISCONNECTED
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    total_operations: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    _latencies: List[float] = field(default_factory=list)

    def record_operation(self, latency_ms: float, success: bool) -> None:
        """Record operation result."""
        self.total_operations += 1
        self.last_used_at = datetime.utcnow()

        if success:
            self.consecutive_failures = 0
            self._latencies.append(latency_ms)
            if len(self._latencies) > 100:
                self._latencies = self._latencies[-100:]
            self.avg_latency_ms = sum(self._latencies) / len(self._latencies)
        else:
            self.consecutive_failures += 1
            self.total_errors += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'consecutive_failures': self.consecutive_failures,
            'total_operations': self.total_operations,
            'total_errors': self.total_errors,
            'avg_latency_ms': round(self.avg_latency_ms, 2),
        }


# =============================================================================
# MT5 CONNECTION POOL
# =============================================================================

class MT5ConnectionPool:
    """
    Thread-safe connection pool for MetaTrader 5.

    Reduces latency by:
    1. Reusing initialized connections
    2. Eliminating repeated login overhead
    3. Parallel operations via multiple connections
    4. Automatic health monitoring

    Architecture:
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                   MT5ConnectionPool                         │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │   get_connection()  ──►  Available Queue  ──►  Connection  │
    │        │                      │                    │        │
    │        │                      ▼                    ▼        │
    │        │              ┌───────────────┐    ┌────────────┐  │
    │   Waits if         │ Connection 1  │    │ MT5 API    │  │
    │   none available   │ Connection 2  │    │ Operations │  │
    │        │           │ Connection 3  │    └────────────┘  │
    │        │           │ Connection 4  │           │        │
    │        ▼           └───────────────┘           │        │
    │                                                 │        │
    │   Returns to pool  ◄────────────────────────────┘        │
    │                                                           │
    └───────────────────────────────────────────────────────────┘
    ```

    Example:
        config = MT5PoolConfig.from_environment()
        pool = MT5ConnectionPool(config)
        pool.start()

        # Get connection from pool
        with pool.get_connection() as conn:
            # Use connection
            result = conn.order_send(request)

        # Connection automatically returned to pool

        pool.stop()
    """

    def __init__(self, config: MT5PoolConfig):
        """
        Initialize connection pool.

        Args:
            config: MT5PoolConfig
        """
        self.config = config
        self._logger = logging.getLogger("performance.mt5_pool")

        # Connection pool
        self._connections: Dict[int, PooledConnection] = {}
        self._available: Queue = Queue()
        self._lock = threading.RLock()

        # State
        self._running = False
        self._initialized = False

        # Health check thread
        self._health_checker: Optional[threading.Thread] = None

        # Stats
        self._stats = {
            'checkouts': 0,
            'checkins': 0,
            'timeouts': 0,
            'reconnections': 0,
        }

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self) -> bool:
        """
        Start the connection pool.

        Returns:
            True if at least one connection was established
        """
        if self._running:
            return True

        if not MT5_AVAILABLE:
            self._logger.error("MetaTrader5 package not installed")
            return False

        self._running = True
        success_count = 0

        # Initialize connections
        for i in range(self.config.pool_size):
            conn = PooledConnection(id=i)
            self._connections[i] = conn

            if self._connect(conn):
                self._available.put(i)
                success_count += 1

        if success_count == 0:
            self._logger.error("Failed to establish any MT5 connections")
            self._running = False
            return False

        self._initialized = True

        # Start health checker
        self._health_checker = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="mt5-health-checker"
        )
        self._health_checker.start()

        self._logger.info(
            f"MT5 pool started: {success_count}/{self.config.pool_size} connections"
        )
        return True

    def stop(self) -> None:
        """Stop the connection pool and close all connections."""
        self._running = False

        # Wait for health checker
        if self._health_checker:
            self._health_checker.join(timeout=5.0)

        # Close all connections
        with self._lock:
            for conn in self._connections.values():
                self._disconnect(conn)

            self._connections.clear()

        # Clear queue
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except Empty:
                break

        # Shutdown MT5
        if MT5_AVAILABLE:
            try:
                mt5.shutdown()
            except Exception:
                pass

        self._initialized = False
        self._logger.info("MT5 pool stopped")

    # =========================================================================
    # CONNECTION MANAGEMENT
    # =========================================================================

    @contextmanager
    def get_connection(self) -> Generator['MT5Connection', None, None]:
        """
        Get a connection from the pool.

        Yields:
            MT5Connection wrapper

        Raises:
            TimeoutError: If no connection available within timeout
        """
        conn_id = None

        try:
            # Get available connection
            try:
                conn_id = self._available.get(
                    timeout=self.config.checkout_timeout_seconds
                )
                self._stats['checkouts'] += 1
            except Empty:
                self._stats['timeouts'] += 1
                raise TimeoutError("No MT5 connection available")

            conn = self._connections.get(conn_id)
            if not conn or conn.state != ConnectionState.CONNECTED:
                # Try to reconnect
                if not self._connect(conn):
                    raise RuntimeError(f"Connection {conn_id} unavailable")

            # Yield wrapped connection
            yield MT5Connection(self, conn)

        finally:
            # Return connection to pool
            if conn_id is not None:
                self._available.put(conn_id)
                self._stats['checkins'] += 1

    def _connect(self, conn: PooledConnection) -> bool:
        """Establish connection to MT5."""
        if not MT5_AVAILABLE:
            return False

        conn.state = ConnectionState.CONNECTING

        for attempt in range(self.config.retry_attempts):
            try:
                # Initialize MT5 (if not already)
                if not mt5.initialize(path=self.config.path or None):
                    error = mt5.last_error()
                    self._logger.warning(f"MT5 initialize failed: {error}")
                    time.sleep(self.config.retry_delay_seconds)
                    continue

                # Login
                if self.config.login > 0:
                    authorized = mt5.login(
                        login=self.config.login,
                        password=self.config.password,
                        server=self.config.server,
                        timeout=int(self.config.connection_timeout_seconds * 1000)
                    )

                    if not authorized:
                        error = mt5.last_error()
                        self._logger.warning(f"MT5 login failed: {error}")
                        time.sleep(self.config.retry_delay_seconds)
                        continue

                conn.state = ConnectionState.CONNECTED
                conn.consecutive_failures = 0
                self._logger.info(f"Connection {conn.id} established")
                return True

            except Exception as e:
                self._logger.error(f"Connection error: {e}")
                time.sleep(self.config.retry_delay_seconds)

        conn.state = ConnectionState.ERROR
        return False

    def _disconnect(self, conn: PooledConnection) -> None:
        """Disconnect a connection."""
        conn.state = ConnectionState.CLOSED
        # Note: MT5 shutdown is global, handled in stop()

    def _reconnect(self, conn: PooledConnection) -> bool:
        """Attempt to reconnect."""
        self._stats['reconnections'] += 1
        return self._connect(conn)

    # =========================================================================
    # HEALTH CHECKING
    # =========================================================================

    def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            time.sleep(self.config.health_check_interval_seconds)

            if not self._running:
                break

            self._perform_health_checks()

    def _perform_health_checks(self) -> None:
        """Check health of all connections."""
        for conn_id, conn in self._connections.items():
            if conn.state == ConnectionState.CONNECTED:
                # Try a simple operation
                try:
                    # Get account info as health check
                    if MT5_AVAILABLE and mt5.account_info() is not None:
                        conn.last_health_check = datetime.utcnow()
                        conn.consecutive_failures = 0
                    else:
                        conn.consecutive_failures += 1

                except Exception as e:
                    self._logger.warning(f"Health check failed for conn {conn_id}: {e}")
                    conn.consecutive_failures += 1

                # Reconnect if too many failures
                if conn.consecutive_failures >= self.config.max_consecutive_failures:
                    self._logger.warning(f"Reconnecting conn {conn_id} after failures")
                    self._reconnect(conn)

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        connections_status = {}
        for conn_id, conn in self._connections.items():
            connections_status[conn_id] = conn.to_dict()

        return {
            **self._stats,
            'pool_size': self.config.pool_size,
            'available': self._available.qsize(),
            'running': self._running,
            'connections': connections_status,
        }

    def get_available_count(self) -> int:
        """Get number of available connections."""
        return self._available.qsize()

    @property
    def is_running(self) -> bool:
        return self._running


# =============================================================================
# MT5 CONNECTION WRAPPER
# =============================================================================

class MT5Connection:
    """
    Wrapper for a pooled MT5 connection.

    Provides the same interface as direct MT5 calls
    but tracks latency and handles errors.
    """

    def __init__(self, pool: MT5ConnectionPool, conn: PooledConnection):
        self._pool = pool
        self._conn = conn
        self._logger = logging.getLogger("performance.mt5_conn")

    def order_send(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send an order."""
        return self._execute_operation(
            lambda: mt5.order_send(self._build_request(request)),
            "order_send"
        )

    def positions_get(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get open positions."""
        def op():
            if symbol:
                return mt5.positions_get(symbol=symbol)
            return mt5.positions_get()

        result = self._execute_operation(op, "positions_get")
        if result is None:
            return []
        return [self._position_to_dict(p) for p in result]

    def symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information."""
        def op():
            info = mt5.symbol_info(symbol)
            return self._symbol_info_to_dict(info) if info else None

        return self._execute_operation(op, "symbol_info")

    def symbol_info_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current tick for symbol."""
        def op():
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume,
                    'time': tick.time,
                }
            return None

        return self._execute_operation(op, "symbol_info_tick")

    def account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        def op():
            info = mt5.account_info()
            if info:
                return {
                    'login': info.login,
                    'balance': info.balance,
                    'equity': info.equity,
                    'margin': info.margin,
                    'margin_free': info.margin_free,
                    'profit': info.profit,
                }
            return None

        return self._execute_operation(op, "account_info")

    def _execute_operation(
        self,
        operation: callable,
        operation_name: str
    ) -> Any:
        """Execute operation with timing and error handling."""
        start = time.perf_counter()

        try:
            if not MT5_AVAILABLE:
                return None

            result = operation()
            success = result is not None

            latency_ms = (time.perf_counter() - start) * 1000
            self._conn.record_operation(latency_ms, success)

            if not success:
                error = mt5.last_error()
                self._logger.warning(f"{operation_name} failed: {error}")

            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            self._conn.record_operation(latency_ms, False)
            self._logger.error(f"{operation_name} error: {e}")
            return None

    def _build_request(self, params: Dict[str, Any]) -> Any:
        """Build MT5 trade request."""
        if not MT5_AVAILABLE:
            return None

        action = mt5.TRADE_ACTION_DEAL

        request = {
            "action": action,
            "symbol": params.get('symbol'),
            "volume": params.get('volume', 0.1),
            "type": mt5.ORDER_TYPE_BUY if params.get('type', 'BUY') == 'BUY' else mt5.ORDER_TYPE_SELL,
            "deviation": params.get('deviation', 20),
            "magic": params.get('magic', 234000),
            "comment": params.get('comment', 'python script'),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if params.get('price'):
            request['price'] = params['price']
        if params.get('sl'):
            request['sl'] = params['sl']
        if params.get('tp'):
            request['tp'] = params['tp']

        return request

    def _position_to_dict(self, pos) -> Dict[str, Any]:
        """Convert MT5 position to dict."""
        return {
            'ticket': pos.ticket,
            'symbol': pos.symbol,
            'type': 'BUY' if pos.type == 0 else 'SELL',
            'volume': pos.volume,
            'price_open': pos.price_open,
            'price_current': pos.price_current,
            'profit': pos.profit,
            'sl': pos.sl,
            'tp': pos.tp,
        }

    def _symbol_info_to_dict(self, info) -> Dict[str, Any]:
        """Convert MT5 symbol info to dict."""
        return {
            'name': info.name,
            'bid': info.bid,
            'ask': info.ask,
            'point': info.point,
            'digits': info.digits,
            'spread': info.spread,
            'trade_mode': info.trade_mode,
        }
