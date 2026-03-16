# =============================================================================
# DEAD MAN'S SWITCH - External Heartbeat Monitoring
# =============================================================================
# Critical safety system that detects when the trading bot stops responding.
#
# Architecture:
#   1. Bot sends heartbeats to external monitoring service
#   2. If heartbeats stop, external service triggers emergency actions:
#      - Alert on-call personnel
#      - Execute emergency position closure via backup API
#      - Log incident for post-mortem
#
# Supported Backends:
#   - HTTP Webhook (custom server, Uptime Robot, etc.)
#   - Redis (for distributed setups)
#   - File-based (for local development)
#   - AWS CloudWatch (for AWS infrastructure)
#
# Usage:
#   switch = DeadManSwitch(config, alert_manager)
#   switch.start()
#
#   # In main loop
#   switch.heartbeat()
#
#   # On graceful shutdown
#   switch.stop()
#
# =============================================================================

import os
import json
import time
import logging
import threading
import socket
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

# Optional imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


# =============================================================================
# CONFIGURATION
# =============================================================================

class HeartbeatBackend(Enum):
    """Available heartbeat backends."""
    HTTP_WEBHOOK = "http_webhook"
    REDIS = "redis"
    FILE = "file"
    CLOUDWATCH = "cloudwatch"
    MULTI = "multi"  # Use multiple backends for redundancy


class HeartbeatStatus(Enum):
    """Status of the heartbeat system."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Some backends failing
    CRITICAL = "critical"  # All backends failing
    STOPPED = "stopped"


@dataclass
class DeadManSwitchConfig:
    """Configuration for Dead Man's Switch."""
    # Enabled backends (use MULTI for redundancy)
    backend: HeartbeatBackend = HeartbeatBackend.HTTP_WEBHOOK

    # Timing
    heartbeat_interval_seconds: int = 30
    timeout_seconds: int = 60  # Time without heartbeat before alarm
    startup_grace_seconds: int = 120  # Grace period during startup

    # HTTP Webhook backend
    webhook_url: str = ""
    webhook_method: str = "POST"
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    webhook_timeout: int = 10

    # Redis backend
    redis_url: str = ""
    redis_key_prefix: str = "trading_bot:heartbeat"
    redis_ttl_seconds: int = 120  # Key expiry in Redis

    # File backend (for local dev)
    heartbeat_file_path: str = ".heartbeat"

    # CloudWatch backend
    cloudwatch_namespace: str = "TradingBot"
    cloudwatch_metric_name: str = "Heartbeat"

    # Bot identification
    bot_id: str = ""
    environment: str = "production"

    # Emergency actions
    enable_emergency_closure: bool = True
    emergency_webhook_url: str = ""  # Called when bot is detected as dead
    emergency_api_credentials: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_environment(cls) -> 'DeadManSwitchConfig':
        """Create config from environment variables."""
        backend_str = os.getenv('HEARTBEAT_BACKEND', 'http_webhook')

        return cls(
            backend=HeartbeatBackend(backend_str),
            heartbeat_interval_seconds=int(os.getenv('HEARTBEAT_INTERVAL', '30')),
            timeout_seconds=int(os.getenv('HEARTBEAT_TIMEOUT', '60')),
            webhook_url=os.getenv('HEARTBEAT_WEBHOOK_URL', ''),
            redis_url=os.getenv('REDIS_URL', ''),
            bot_id=os.getenv('BOT_ID', socket.gethostname()),
            environment=os.getenv('ENVIRONMENT', 'production'),
            emergency_webhook_url=os.getenv('EMERGENCY_WEBHOOK_URL', ''),
        )


# =============================================================================
# HEARTBEAT DATA
# =============================================================================

@dataclass
class HeartbeatPayload:
    """Data sent with each heartbeat."""
    bot_id: str
    timestamp: datetime
    environment: str
    status: str = "alive"
    uptime_seconds: float = 0.0
    positions_count: int = 0
    last_trade_at: Optional[datetime] = None
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'bot_id': self.bot_id,
            'timestamp': self.timestamp.isoformat(),
            'environment': self.environment,
            'status': self.status,
            'uptime_seconds': self.uptime_seconds,
            'positions_count': self.positions_count,
            'last_trade_at': self.last_trade_at.isoformat() if self.last_trade_at else None,
            'memory_mb': self.memory_mb,
            'cpu_percent': self.cpu_percent,
            'metadata': self.metadata,
        }


# =============================================================================
# DEAD MAN'S SWITCH
# =============================================================================

class DeadManSwitch:
    """
    Critical safety system for detecting and responding to bot failures.

    The Dead Man's Switch sends regular heartbeats to external monitoring.
    If heartbeats stop (due to crash, hang, network issues), the external
    monitor triggers emergency actions.

    Architecture:
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    TRADING BOT PROCESS                      │
    │                                                             │
    │   ┌─────────────┐    heartbeat()    ┌───────────────────┐  │
    │   │ Main Loop   │ ─────────────────►│ DeadManSwitch     │  │
    │   └─────────────┘                   │ (sends heartbeats)│  │
    │                                     └─────────┬─────────┘  │
    └───────────────────────────────────────────────┼─────────────┘
                                                    │
                                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │               EXTERNAL MONITORING SERVICE                   │
    │                                                             │
    │   ┌─────────────────────────────────────────────────────┐  │
    │   │ Receives heartbeats, tracks last_seen timestamp     │  │
    │   │                                                     │  │
    │   │ IF no heartbeat for > timeout:                      │  │
    │   │   1. Send CRITICAL alert (PagerDuty, SMS)           │  │
    │   │   2. Call emergency webhook                         │  │
    │   │   3. Close all positions via backup API             │  │
    │   │   4. Log incident                                   │  │
    │   └─────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
    ```

    Example:
        config = DeadManSwitchConfig.from_environment()
        switch = DeadManSwitch(config, alert_manager)

        # Start heartbeat thread
        switch.start()

        # In main trading loop
        while running:
            # ... trading logic ...
            switch.heartbeat(positions_count=len(positions))

        # Graceful shutdown
        switch.stop()
    """

    def __init__(
        self,
        config: DeadManSwitchConfig,
        alert_manager=None,
        on_failure_callback: Optional[Callable] = None
    ):
        """
        Initialize Dead Man's Switch.

        Args:
            config: DeadManSwitchConfig
            alert_manager: Optional AlertManager for local alerts
            on_failure_callback: Called if heartbeat sending fails repeatedly
        """
        self.config = config
        self.alert_manager = alert_manager
        self.on_failure_callback = on_failure_callback
        self._logger = logging.getLogger("security.deadman")

        # State
        self._status = HeartbeatStatus.INITIALIZING
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Timing
        self._start_time = datetime.utcnow()
        self._last_heartbeat: Optional[datetime] = None
        self._last_successful_heartbeat: Optional[datetime] = None
        self._consecutive_failures = 0
        self._max_consecutive_failures = 3

        # Context data (updated by caller)
        self._positions_count = 0
        self._last_trade_at: Optional[datetime] = None
        self._custom_metadata: Dict[str, Any] = {}

        # Backend clients
        self._redis_client = None
        self._init_backends()

        # Stats
        self._stats = {
            'heartbeats_sent': 0,
            'heartbeats_failed': 0,
            'uptime_seconds': 0,
        }

    def _init_backends(self) -> None:
        """Initialize backend connections."""
        if self.config.backend in [HeartbeatBackend.REDIS, HeartbeatBackend.MULTI]:
            if REDIS_AVAILABLE and self.config.redis_url:
                try:
                    self._redis_client = redis.from_url(self.config.redis_url)
                    self._redis_client.ping()
                    self._logger.info("Redis heartbeat backend initialized")
                except Exception as e:
                    self._logger.warning(f"Failed to connect to Redis: {e}")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def start(self) -> None:
        """Start the heartbeat thread."""
        if self._running:
            self._logger.warning("DeadManSwitch already running")
            return

        self._running = True
        self._start_time = datetime.utcnow()
        self._status = HeartbeatStatus.HEALTHY

        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="deadman-heartbeat"
        )
        self._thread.start()

        # Send initial heartbeat
        self._send_heartbeat()

        self._logger.info(
            f"DeadManSwitch started: interval={self.config.heartbeat_interval_seconds}s, "
            f"timeout={self.config.timeout_seconds}s"
        )

    def stop(self, graceful: bool = True) -> None:
        """
        Stop the heartbeat thread.

        Args:
            graceful: If True, send final "stopping" heartbeat
        """
        self._running = False
        self._status = HeartbeatStatus.STOPPED

        if graceful:
            # Send final heartbeat indicating graceful shutdown
            self._send_heartbeat(status="stopping")
            self._logger.info("Sent graceful shutdown heartbeat")

        if self._thread:
            self._thread.join(timeout=5)

        self._logger.info("DeadManSwitch stopped")

    def heartbeat(
        self,
        positions_count: int = 0,
        last_trade_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update heartbeat context (called from main trading loop).

        This updates the metadata sent with automatic heartbeats.
        You don't need to call this on every tick - the background
        thread handles the actual sending.

        Args:
            positions_count: Current number of open positions
            last_trade_at: Timestamp of last trade
            metadata: Additional context data
        """
        with self._lock:
            self._positions_count = positions_count
            self._last_trade_at = last_trade_at
            if metadata:
                self._custom_metadata.update(metadata)

    def force_heartbeat(self) -> bool:
        """
        Force an immediate heartbeat (outside normal interval).

        Returns:
            True if heartbeat was sent successfully
        """
        return self._send_heartbeat()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the Dead Man's Switch."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds()

        return {
            'status': self._status.value,
            'running': self._running,
            'uptime_seconds': uptime,
            'last_heartbeat': self._last_heartbeat.isoformat() if self._last_heartbeat else None,
            'last_successful': self._last_successful_heartbeat.isoformat() if self._last_successful_heartbeat else None,
            'consecutive_failures': self._consecutive_failures,
            'stats': self._stats,
            'config': {
                'interval': self.config.heartbeat_interval_seconds,
                'timeout': self.config.timeout_seconds,
                'backend': self.config.backend.value,
            }
        }

    @contextmanager
    def session(self):
        """
        Context manager for automatic start/stop.

        Example:
            with dead_man_switch.session():
                # Trading loop
                while running:
                    switch.heartbeat()
        """
        try:
            self.start()
            yield self
        finally:
            self.stop(graceful=True)

    # =========================================================================
    # HEARTBEAT LOGIC
    # =========================================================================

    def _heartbeat_loop(self) -> None:
        """Background thread that sends periodic heartbeats."""
        while self._running:
            try:
                self._send_heartbeat()
                time.sleep(self.config.heartbeat_interval_seconds)
            except Exception as e:
                self._logger.error(f"Heartbeat loop error: {e}")
                time.sleep(5)  # Brief pause on error

    def _send_heartbeat(self, status: str = "alive") -> bool:
        """Send heartbeat to configured backends."""
        with self._lock:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()

            # Build payload
            payload = HeartbeatPayload(
                bot_id=self.config.bot_id,
                timestamp=datetime.utcnow(),
                environment=self.config.environment,
                status=status,
                uptime_seconds=uptime,
                positions_count=self._positions_count,
                last_trade_at=self._last_trade_at,
                memory_mb=self._get_memory_usage(),
                cpu_percent=self._get_cpu_usage(),
                metadata=self._custom_metadata,
            )

            self._last_heartbeat = datetime.utcnow()
            self._stats['uptime_seconds'] = uptime

        # Send to backend(s)
        success = False

        if self.config.backend == HeartbeatBackend.HTTP_WEBHOOK:
            success = self._send_http_heartbeat(payload)
        elif self.config.backend == HeartbeatBackend.REDIS:
            success = self._send_redis_heartbeat(payload)
        elif self.config.backend == HeartbeatBackend.FILE:
            success = self._send_file_heartbeat(payload)
        elif self.config.backend == HeartbeatBackend.MULTI:
            # Try all backends, succeed if any works
            results = [
                self._send_http_heartbeat(payload),
                self._send_redis_heartbeat(payload),
                self._send_file_heartbeat(payload),
            ]
            success = any(results)

        # Update state
        if success:
            self._stats['heartbeats_sent'] += 1
            self._last_successful_heartbeat = datetime.utcnow()
            self._consecutive_failures = 0
            if self._status != HeartbeatStatus.STOPPED:
                self._status = HeartbeatStatus.HEALTHY
        else:
            self._stats['heartbeats_failed'] += 1
            self._consecutive_failures += 1

            if self._consecutive_failures >= self._max_consecutive_failures:
                self._status = HeartbeatStatus.CRITICAL
                self._handle_heartbeat_failure()

        return success

    def _send_http_heartbeat(self, payload: HeartbeatPayload) -> bool:
        """Send heartbeat via HTTP webhook."""
        if not REQUESTS_AVAILABLE:
            return False

        if not self.config.webhook_url:
            return False

        try:
            if self.config.webhook_method.upper() == "POST":
                response = requests.post(
                    self.config.webhook_url,
                    json=payload.to_dict(),
                    headers={
                        "Content-Type": "application/json",
                        "X-Bot-ID": self.config.bot_id,
                        **self.config.webhook_headers,
                    },
                    timeout=self.config.webhook_timeout,
                )
            else:
                response = requests.get(
                    self.config.webhook_url,
                    params={'bot_id': self.config.bot_id, 'status': payload.status},
                    headers=self.config.webhook_headers,
                    timeout=self.config.webhook_timeout,
                )

            return 200 <= response.status_code < 300

        except Exception as e:
            self._logger.warning(f"HTTP heartbeat failed: {e}")
            return False

    def _send_redis_heartbeat(self, payload: HeartbeatPayload) -> bool:
        """Send heartbeat via Redis."""
        if not self._redis_client:
            return False

        try:
            key = f"{self.config.redis_key_prefix}:{self.config.bot_id}"
            self._redis_client.setex(
                key,
                self.config.redis_ttl_seconds,
                json.dumps(payload.to_dict())
            )

            # Also publish to channel for real-time monitoring
            self._redis_client.publish(
                f"{self.config.redis_key_prefix}:events",
                json.dumps(payload.to_dict())
            )

            return True

        except Exception as e:
            self._logger.warning(f"Redis heartbeat failed: {e}")
            return False

    def _send_file_heartbeat(self, payload: HeartbeatPayload) -> bool:
        """Send heartbeat via file (for local development)."""
        try:
            file_path = Path(self.config.heartbeat_file_path)
            file_path.write_text(json.dumps(payload.to_dict(), indent=2))
            return True

        except Exception as e:
            self._logger.warning(f"File heartbeat failed: {e}")
            return False

    # =========================================================================
    # FAILURE HANDLING
    # =========================================================================

    def _handle_heartbeat_failure(self) -> None:
        """Handle repeated heartbeat failures."""
        self._logger.critical(
            f"Heartbeat failure: {self._consecutive_failures} consecutive failures"
        )

        # Send local alert if available
        if self.alert_manager:
            self.alert_manager.critical(
                "Dead Man's Switch Failure",
                message=f"Unable to send heartbeats for {self._consecutive_failures} attempts",
                details={
                    'bot_id': self.config.bot_id,
                    'consecutive_failures': self._consecutive_failures,
                    'last_successful': self._last_successful_heartbeat.isoformat() if self._last_successful_heartbeat else None,
                },
                source="dead_man_switch"
            )

        # Call failure callback
        if self.on_failure_callback:
            try:
                self.on_failure_callback()
            except Exception as e:
                self._logger.error(f"Failure callback error: {e}")

    # =========================================================================
    # SYSTEM METRICS
    # =========================================================================

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0


# =============================================================================
# EXTERNAL MONITOR (Server-Side Component)
# =============================================================================

class HeartbeatMonitor:
    """
    Server-side component that monitors heartbeats and triggers actions.

    This should run on a SEPARATE server/service (e.g., AWS Lambda,
    separate VPS, monitoring service) to detect when the trading bot dies.

    Example deployment:
        - AWS Lambda triggered every minute by CloudWatch Events
        - Checks Redis/DynamoDB for last heartbeat
        - If stale, triggers emergency closure

    Example:
        monitor = HeartbeatMonitor(config)

        # Called by cron/lambda every minute
        if monitor.check_bot_health("trading-bot-1"):
            print("Bot is healthy")
        else:
            print("Bot is DEAD, triggering emergency actions")
            monitor.trigger_emergency_closure("trading-bot-1")
    """

    def __init__(self, config: DeadManSwitchConfig):
        self.config = config
        self._logger = logging.getLogger("security.monitor")
        self._redis_client = None

        if REDIS_AVAILABLE and config.redis_url:
            self._redis_client = redis.from_url(config.redis_url)

    def check_bot_health(self, bot_id: str) -> bool:
        """
        Check if a bot is healthy (sending heartbeats).

        Args:
            bot_id: Bot identifier to check

        Returns:
            True if bot is healthy, False if dead/unresponsive
        """
        if self._redis_client:
            return self._check_redis_heartbeat(bot_id)
        else:
            return self._check_file_heartbeat(bot_id)

    def _check_redis_heartbeat(self, bot_id: str) -> bool:
        """Check heartbeat in Redis."""
        try:
            key = f"{self.config.redis_key_prefix}:{bot_id}"
            data = self._redis_client.get(key)

            if not data:
                self._logger.warning(f"No heartbeat found for {bot_id}")
                return False

            heartbeat = json.loads(data)
            timestamp = datetime.fromisoformat(heartbeat['timestamp'])
            age = (datetime.utcnow() - timestamp).total_seconds()

            if age > self.config.timeout_seconds:
                self._logger.warning(
                    f"Stale heartbeat for {bot_id}: {age:.0f}s old"
                )
                return False

            return True

        except Exception as e:
            self._logger.error(f"Error checking Redis heartbeat: {e}")
            return False

    def _check_file_heartbeat(self, bot_id: str) -> bool:
        """Check heartbeat in file."""
        try:
            file_path = Path(self.config.heartbeat_file_path)
            if not file_path.exists():
                return False

            data = json.loads(file_path.read_text())
            timestamp = datetime.fromisoformat(data['timestamp'])
            age = (datetime.utcnow() - timestamp).total_seconds()

            return age <= self.config.timeout_seconds

        except Exception as e:
            self._logger.error(f"Error checking file heartbeat: {e}")
            return False

    def trigger_emergency_closure(self, bot_id: str) -> bool:
        """
        Trigger emergency position closure for a dead bot.

        This calls the emergency webhook which should:
        1. Close all positions via broker API
        2. Send alerts
        3. Log the incident

        Args:
            bot_id: Bot that needs emergency closure

        Returns:
            True if emergency action was triggered
        """
        if not self.config.emergency_webhook_url:
            self._logger.error("No emergency webhook configured!")
            return False

        try:
            response = requests.post(
                self.config.emergency_webhook_url,
                json={
                    'action': 'emergency_closure',
                    'bot_id': bot_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'reason': 'heartbeat_timeout',
                },
                headers={
                    "Content-Type": "application/json",
                    **self.config.emergency_api_credentials,
                },
                timeout=30,
            )

            self._logger.critical(
                f"Emergency closure triggered for {bot_id}: {response.status_code}"
            )

            return 200 <= response.status_code < 300

        except Exception as e:
            self._logger.critical(f"Failed to trigger emergency closure: {e}")
            return False
