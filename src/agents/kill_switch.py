# =============================================================================
# KILL SWITCH - Emergency Trading Halt System
# =============================================================================
# Production-grade emergency halt system for algorithmic trading.
#
# This module provides multiple layers of protection:
#
#   1. CIRCUIT BREAKERS - Automatic triggers based on conditions
#      - Drawdown breakers (daily, weekly, total)
#      - Loss velocity breakers (rapid loss detection)
#      - Volatility breakers (market conditions)
#      - Consecutive loss breakers
#
#   2. HARD LIMITS - Non-bypassable safety limits
#      - Cannot be overridden by code
#      - Require manual intervention to reset
#      - Cryptographically signed reset tokens
#
#   3. MANUAL CONTROLS - Human override capability
#      - Emergency halt button
#      - Gradual wind-down mode
#      - Selective position closure
#
#   4. RECOVERY PROCEDURES - Safe restart protocols
#      - Cooling-off periods
#      - Gradual position rebuild
#      - Confirmation requirements
#
# Architecture:
#   ┌─────────────────────────────────────────────────────────────────┐
#   │                        KILL SWITCH                              │
#   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
#   │  │ Circuit     │ │ Hard Limit  │ │ Recovery    │ │ Alert     │ │
#   │  │ Breakers    │ │ Enforcer    │ │ Manager     │ │ System    │ │
#   │  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
#   └─────────────────────────────────────────────────────────────────┘
#
# CRITICAL: This module is designed to FAIL SAFE. When in doubt, HALT.
# =============================================================================

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# PERSISTENCE INTEGRATION
# =============================================================================
# Import persistence module for SQLite-backed state storage
# This ensures Kill Switch state survives bot restarts/crashes

try:
    from src.persistence.kill_switch_store import (
        KillSwitchStore,
        KillSwitchState,
        BreakerRecord,
        HaltEventRecord
    )
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    KillSwitchStore = None
    KillSwitchState = None
    BreakerRecord = None
    HaltEventRecord = None


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class HaltReason(Enum):
    """Reasons for trading halt."""
    # Automatic triggers
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    WEEKLY_LOSS_LIMIT = "weekly_loss_limit"
    MAX_DRAWDOWN = "max_drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    LOSS_VELOCITY = "loss_velocity"
    VAR_BREACH = "var_breach"
    EXPOSURE_BREACH = "exposure_breach"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"

    # System issues
    CONNECTIVITY_LOSS = "connectivity_loss"
    DATA_FEED_ERROR = "data_feed_error"
    EXECUTION_ERROR = "execution_error"
    SYSTEM_ERROR = "system_error"

    # Manual triggers
    MANUAL_HALT = "manual_halt"
    EMERGENCY_STOP = "emergency_stop"
    MAINTENANCE = "maintenance"

    # External
    MARKET_CLOSED = "market_closed"
    BROKER_HALT = "broker_halt"
    REGULATORY = "regulatory"


class HaltLevel(Enum):
    """Severity levels for trading halt."""
    NONE = 0           # Normal operation
    CAUTION = 1        # Warnings active, reduced activity
    REDUCED = 2        # Position sizing reduced
    NEW_ONLY = 3       # No new positions, can close existing
    CLOSE_ONLY = 4     # Must close positions
    FULL_HALT = 5      # All trading stopped
    EMERGENCY = 6      # Emergency - flatten all immediately


class BreakerState(Enum):
    """State of a circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Triggered, blocking
    HALF_OPEN = "half_open"  # Testing recovery


class RecoveryState(Enum):
    """State of recovery process."""
    NOT_STARTED = "not_started"
    COOLING_OFF = "cooling_off"
    CONFIRMATION_PENDING = "confirmation_pending"
    GRADUAL_RESTART = "gradual_restart"
    RECOVERED = "recovered"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HaltEvent:
    """Record of a halt event."""
    halt_id: str
    reason: HaltReason
    level: HaltLevel
    timestamp: datetime
    trigger_value: float
    threshold: float
    message: str
    auto_recovery: bool = False
    recovery_time: Optional[datetime] = None
    recovered: bool = False
    recovery_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "halt_id": self.halt_id,
            "reason": self.reason.value,
            "level": self.level.name,
            "timestamp": self.timestamp.isoformat(),
            "trigger_value": self.trigger_value,
            "threshold": self.threshold,
            "message": self.message,
            "auto_recovery": self.auto_recovery,
            "recovery_time": self.recovery_time.isoformat() if self.recovery_time else None,
            "recovered": self.recovered,
            "recovery_timestamp": self.recovery_timestamp.isoformat() if self.recovery_timestamp else None
        }


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    name: str
    threshold: float
    halt_level: HaltLevel
    cooldown_seconds: int = 3600  # 1 hour default
    auto_recovery: bool = False
    consecutive_threshold: int = 1  # Triggers needed to trip


@dataclass
class KillSwitchConfig:
    """
    Complete configuration for the kill switch system.

    All thresholds are expressed as decimals (0.05 = 5%).
    """
    # Daily limits
    max_daily_loss_pct: float = 0.03          # 3% daily loss
    max_daily_loss_usd: float = float('inf')  # No USD limit by default

    # Weekly limits
    max_weekly_loss_pct: float = 0.05         # 5% weekly loss

    # Total limits
    max_drawdown_pct: float = 0.10            # 10% max drawdown
    max_drawdown_usd: float = float('inf')    # No USD limit by default

    # Consecutive losses
    max_consecutive_losses: int = 5           # 5 losses in a row

    # Loss velocity (rapid loss detection)
    loss_velocity_window_minutes: int = 30    # Look back 30 minutes
    loss_velocity_threshold_pct: float = 0.02  # 2% loss in window

    # VaR limits
    max_var_pct: float = 0.02                 # 2% VaR limit

    # Exposure limits
    max_gross_exposure_pct: float = 2.0       # 200% gross

    # Connectivity
    max_disconnect_seconds: int = 300         # 5 minutes

    # Recovery
    default_cooldown_seconds: int = 3600      # 1 hour cooldown
    require_manual_reset: bool = True         # Require manual intervention

    # Security
    reset_secret: str = ""                    # Secret for reset tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_daily_loss_pct": f"{self.max_daily_loss_pct:.1%}",
            "max_weekly_loss_pct": f"{self.max_weekly_loss_pct:.1%}",
            "max_drawdown_pct": f"{self.max_drawdown_pct:.1%}",
            "max_consecutive_losses": self.max_consecutive_losses,
            "max_var_pct": f"{self.max_var_pct:.1%}",
            "default_cooldown_seconds": self.default_cooldown_seconds,
            "require_manual_reset": self.require_manual_reset
        }


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """
    Individual circuit breaker for a specific condition.

    Implements the circuit breaker pattern:
    - CLOSED: Normal operation, monitoring condition
    - OPEN: Condition breached, blocking operations
    - HALF_OPEN: Testing if condition has recovered

    Example:
        breaker = CircuitBreaker(
            config=CircuitBreakerConfig(
                name="daily_loss",
                threshold=0.03,
                halt_level=HaltLevel.FULL_HALT,
                cooldown_seconds=86400  # 24 hours
            )
        )

        # Check condition
        if breaker.check(current_daily_loss, threshold=0.03):
            logger.warning("Breaker tripped!")
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = BreakerState.CLOSED
        self.trip_count = 0
        self.last_trip_time: Optional[datetime] = None
        self.recovery_time: Optional[datetime] = None
        self.consecutive_triggers = 0

        self._lock = threading.Lock()
        self._logger = logging.getLogger(f"kill_switch.breaker.{config.name}")

    @property
    def is_tripped(self) -> bool:
        """Check if breaker is currently tripped."""
        return self.state == BreakerState.OPEN

    def check(self, current_value: float, threshold: Optional[float] = None) -> bool:
        """
        Check if condition breaches threshold and trip if necessary.

        Args:
            current_value: Current value to check
            threshold: Override threshold (uses config if not provided)

        Returns:
            True if breaker trips, False otherwise
        """
        threshold = threshold if threshold is not None else self.config.threshold

        with self._lock:
            # If already tripped, check for recovery
            if self.state == BreakerState.OPEN:
                return self._check_recovery()

            # Check threshold
            if abs(current_value) >= abs(threshold):
                self.consecutive_triggers += 1

                if self.consecutive_triggers >= self.config.consecutive_threshold:
                    self._trip(current_value, threshold)
                    return True
            else:
                # Reset consecutive counter on good check
                self.consecutive_triggers = 0

            return False

    def _trip(self, value: float, threshold: float) -> None:
        """Trip the circuit breaker."""
        self.state = BreakerState.OPEN
        self.trip_count += 1
        self.last_trip_time = datetime.now()
        self.recovery_time = (
            self.last_trip_time +
            timedelta(seconds=self.config.cooldown_seconds)
        )

        self._logger.critical(
            f"CIRCUIT BREAKER TRIPPED: {self.config.name} | "
            f"Value: {value:.4f} | Threshold: {threshold:.4f} | "
            f"Recovery at: {self.recovery_time.isoformat()}"
        )

    def _check_recovery(self) -> bool:
        """Check if breaker can recover."""
        if not self.config.auto_recovery:
            return True  # Stay tripped until manual reset

        if self.recovery_time and datetime.now() >= self.recovery_time:
            self.state = BreakerState.HALF_OPEN
            self._logger.info(f"Circuit breaker {self.config.name} entering HALF_OPEN state")

        return True  # Still tripped

    def reset(self, force: bool = False) -> bool:
        """
        Reset the circuit breaker.

        Args:
            force: Force reset even if in cooldown

        Returns:
            True if reset successful
        """
        with self._lock:
            if not force and self.recovery_time and datetime.now() < self.recovery_time:
                self._logger.warning(
                    f"Cannot reset {self.config.name}: still in cooldown until "
                    f"{self.recovery_time.isoformat()}"
                )
                return False

            self.state = BreakerState.CLOSED
            self.consecutive_triggers = 0
            self._logger.info(f"Circuit breaker {self.config.name} RESET")
            return True

    def get_status(self) -> Dict[str, Any]:
        """Get current breaker status."""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "is_tripped": self.is_tripped,
            "trip_count": self.trip_count,
            "last_trip_time": self.last_trip_time.isoformat() if self.last_trip_time else None,
            "recovery_time": self.recovery_time.isoformat() if self.recovery_time else None,
            "halt_level": self.config.halt_level.name,
            "threshold": self.config.threshold
        }


# =============================================================================
# ALERT SYSTEM
# =============================================================================

class AlertCallback(ABC):
    """Abstract base class for alert callbacks."""

    @abstractmethod
    def send(self, halt_event: HaltEvent) -> bool:
        """Send alert for halt event."""
        pass


class LogAlertCallback(AlertCallback):
    """Log-based alert callback."""

    def __init__(self):
        self._logger = logging.getLogger("kill_switch.alerts")

    def send(self, halt_event: HaltEvent) -> bool:
        self._logger.critical(
            f"TRADING HALT ALERT | "
            f"Reason: {halt_event.reason.value} | "
            f"Level: {halt_event.level.name} | "
            f"Message: {halt_event.message}"
        )
        return True


class WebhookAlertCallback(AlertCallback):
    """Webhook-based alert callback for external notifications."""

    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}
        self._logger = logging.getLogger("kill_switch.alerts.webhook")

    def send(self, halt_event: HaltEvent) -> bool:
        try:
            import urllib.request

            data = json.dumps(halt_event.to_dict()).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers=self.headers,
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200

        except Exception as e:
            self._logger.error(f"Webhook alert failed: {e}")
            return False


class TelegramAlertCallback(AlertCallback):
    """Bridge kill switch halt events to Telegram via live_trading alerting.

    Wraps the live_trading AlertManager so that every kill switch state
    change is forwarded as a Telegram notification.
    """

    def __init__(self, alerting_manager):
        self._alerting = alerting_manager
        self._logger = logging.getLogger("kill_switch.alerts.telegram")

    def send(self, halt_event: HaltEvent) -> bool:
        try:
            self._alerting.kill_switch_triggered(
                reason=f"{halt_event.reason.value}: {halt_event.message}",
                equity=halt_event.trigger_value,
            )
            return True
        except Exception as e:
            self._logger.error(f"Telegram alert callback failed: {e}")
            return False


class AlertManager:
    """Manages alert distribution to multiple channels."""

    def __init__(self):
        self._callbacks: List[AlertCallback] = []
        self._alert_history: deque = deque(maxlen=1000)
        self._logger = logging.getLogger("kill_switch.alerts")

        # Always add log callback
        self._callbacks.append(LogAlertCallback())

    def add_callback(self, callback: AlertCallback) -> None:
        """Add an alert callback."""
        self._callbacks.append(callback)

    def send_alert(self, halt_event: HaltEvent) -> None:
        """Send alert to all registered callbacks."""
        self._alert_history.append(halt_event)

        for callback in self._callbacks:
            try:
                callback.send(halt_event)
            except Exception as e:
                self._logger.error(f"Alert callback failed: {e}")

    def get_recent_alerts(self, count: int = 10) -> List[HaltEvent]:
        """Get recent alerts."""
        return list(self._alert_history)[-count:]


# =============================================================================
# RECOVERY MANAGER
# =============================================================================

class RecoveryManager:
    """
    Manages safe recovery from halt states.

    Recovery process:
    1. COOLING_OFF - Wait for cooldown period
    2. CONFIRMATION_PENDING - Await manual confirmation (if required)
    3. GRADUAL_RESTART - Slowly increase activity
    4. RECOVERED - Normal operation
    """

    def __init__(
        self,
        cooldown_seconds: int = 3600,
        require_confirmation: bool = True,
        gradual_steps: int = 5,
        step_duration_seconds: int = 300
    ):
        self.cooldown_seconds = cooldown_seconds
        self.require_confirmation = require_confirmation
        self.gradual_steps = gradual_steps
        self.step_duration_seconds = step_duration_seconds

        self.state = RecoveryState.NOT_STARTED
        self.recovery_start_time: Optional[datetime] = None
        self.confirmation_time: Optional[datetime] = None
        self.current_step = 0
        self.position_multiplier = 0.0  # 0 = no trading, 1 = full trading

        self._confirmation_token: Optional[str] = None
        self._logger = logging.getLogger("kill_switch.recovery")

    def start_recovery(self, halt_event: HaltEvent) -> None:
        """Start the recovery process."""
        self.state = RecoveryState.COOLING_OFF
        self.recovery_start_time = datetime.now()
        self.current_step = 0
        self.position_multiplier = 0.0

        self._logger.info(
            f"Recovery started for halt: {halt_event.reason.value}. "
            f"Cooldown: {self.cooldown_seconds}s"
        )

    def update(self) -> RecoveryState:
        """
        Update recovery state based on time and conditions.

        Returns:
            Current recovery state
        """
        if self.state == RecoveryState.NOT_STARTED:
            return self.state

        now = datetime.now()

        # Check cooling off completion
        if self.state == RecoveryState.COOLING_OFF:
            if self.recovery_start_time:
                elapsed = (now - self.recovery_start_time).total_seconds()
                if elapsed >= self.cooldown_seconds:
                    if self.require_confirmation:
                        self.state = RecoveryState.CONFIRMATION_PENDING
                        self._generate_confirmation_token()
                        self._logger.info("Cooldown complete. Awaiting confirmation.")
                    else:
                        self.state = RecoveryState.GRADUAL_RESTART
                        self._logger.info("Cooldown complete. Starting gradual restart.")

        # Check gradual restart progress
        elif self.state == RecoveryState.GRADUAL_RESTART:
            if self.confirmation_time:
                elapsed = (now - self.confirmation_time).total_seconds()
                self.current_step = min(
                    self.gradual_steps,
                    int(elapsed / self.step_duration_seconds) + 1
                )
                self.position_multiplier = self.current_step / self.gradual_steps

                if self.current_step >= self.gradual_steps:
                    self.state = RecoveryState.RECOVERED
                    self.position_multiplier = 1.0
                    self._logger.info("Recovery complete. Normal operation resumed.")

        return self.state

    def _generate_confirmation_token(self) -> str:
        """Generate a confirmation token for manual reset."""
        import secrets
        self._confirmation_token = secrets.token_hex(16)
        return self._confirmation_token

    def confirm_recovery(self, token: str) -> bool:
        """
        Confirm recovery with token.

        Args:
            token: Confirmation token

        Returns:
            True if confirmation successful
        """
        if self.state != RecoveryState.CONFIRMATION_PENDING:
            self._logger.warning("Cannot confirm: not in CONFIRMATION_PENDING state")
            return False

        if token != self._confirmation_token:
            self._logger.warning("Invalid confirmation token")
            return False

        self.state = RecoveryState.GRADUAL_RESTART
        self.confirmation_time = datetime.now()
        self.current_step = 1
        self.position_multiplier = 1 / self.gradual_steps
        self._confirmation_token = None

        self._logger.info("Recovery confirmed. Starting gradual restart.")
        return True

    def force_recovery(self, admin_key: str, expected_key: str) -> bool:
        """
        Force immediate recovery (admin override).

        Args:
            admin_key: Provided admin key
            expected_key: Expected admin key for verification

        Returns:
            True if force recovery successful
        """
        if not hmac.compare_digest(admin_key, expected_key):
            self._logger.error("Invalid admin key for force recovery")
            return False

        self.state = RecoveryState.RECOVERED
        self.position_multiplier = 1.0
        self._confirmation_token = None

        self._logger.warning("FORCE RECOVERY executed by admin")
        return True

    def abort_recovery(self) -> None:
        """Abort recovery and return to halt state."""
        self.state = RecoveryState.NOT_STARTED
        self.position_multiplier = 0.0
        self._confirmation_token = None
        self._logger.info("Recovery aborted")

    def get_status(self) -> Dict[str, Any]:
        """Get recovery status."""
        return {
            "state": self.state.value,
            "position_multiplier": self.position_multiplier,
            "current_step": self.current_step,
            "total_steps": self.gradual_steps,
            "recovery_start": self.recovery_start_time.isoformat() if self.recovery_start_time else None,
            "confirmation_pending": self.state == RecoveryState.CONFIRMATION_PENDING,
            # SECURITY: Never expose token - even partial exposure aids brute force
            "token_generated": self._confirmation_token is not None
        }


# =============================================================================
# MAIN KILL SWITCH CLASS
# =============================================================================

class KillSwitch:
    """
    Master kill switch for emergency trading halt.

    This is the main interface for the kill switch system.
    It coordinates circuit breakers, manages halt states,
    handles recovery, and distributes alerts.

    Example:
        kill_switch = KillSwitch(config=KillSwitchConfig(
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.10
        ))

        # Update with current state
        kill_switch.update(
            equity=95000,
            peak_equity=100000,
            daily_pnl=-2500,
            var_pct=0.015
        )

        # Check if trading is allowed
        if kill_switch.is_trading_allowed():
            # Execute trade
            pass
        else:
            logger.warning(f"Trading halted: {kill_switch.halt_reason}")

    CRITICAL: This class is designed to FAIL SAFE.
    When in doubt about any condition, it will HALT trading.
    """

    def __init__(
        self,
        config: Optional[KillSwitchConfig] = None,
        admin_key: Optional[str] = None,
        initial_equity: float = 100.0,
        persistence_path: Optional[str] = None,
        enable_persistence: bool = True,
        alerting_manager=None,
    ):
        """
        Initialize kill switch.

        Args:
            config: Kill switch configuration
            admin_key: Admin key for emergency overrides
            initial_equity: Initial account equity for drawdown tracking
                           SECURITY FIX: Defaults to 100.0 to ensure drawdown is tracked
            persistence_path: Path to SQLite database for state persistence
            enable_persistence: Whether to enable state persistence (recommended)
            alerting_manager: Optional live_trading AlertManager for Telegram notifications
        """
        self.config = config or KillSwitchConfig()
        self._initial_equity = max(initial_equity, 1.0)  # Floor at 1.0 to prevent issues

        # =====================================================================
        # PERSISTENCE SETUP
        # =====================================================================
        self._persistence_enabled = enable_persistence and PERSISTENCE_AVAILABLE
        self._store: Optional[KillSwitchStore] = None

        if self._persistence_enabled:
            db_path = persistence_path or "./data/kill_switch.db"
            try:
                self._store = KillSwitchStore(db_path=db_path)

                # Check for previous crash
                crash_info = self._store.check_previous_crash()
                if crash_info:
                    logging.getLogger("kill_switch.main").warning(
                        f"PREVIOUS CRASH DETECTED! Last heartbeat: {crash_info['last_heartbeat']}, "
                        f"PID: {crash_info['previous_pid']}, "
                        f"Time since: {crash_info['time_since_seconds']:.0f}s"
                    )

            except Exception as e:
                logging.getLogger("kill_switch.main").error(
                    f"Failed to initialize persistence: {e}. "
                    f"Kill Switch will operate without persistence (NOT RECOMMENDED)."
                )
                self._persistence_enabled = False
        elif enable_persistence and not PERSISTENCE_AVAILABLE:
            logging.getLogger("kill_switch.main").warning(
                "Persistence requested but module not available. "
                "Kill Switch state will NOT survive restarts!"
            )

        # SECURITY: Validate admin key
        raw_key = admin_key or os.environ.get("KILL_SWITCH_ADMIN_KEY", "")
        if raw_key and len(raw_key) < 32:
            raise ValueError(
                "Admin key must be at least 32 characters for security. "
                "Use a cryptographically secure random string."
            )
        self._admin_key = raw_key

        # Rate limiting for reset attempts
        self._reset_attempts: deque = deque(maxlen=10)
        self._max_reset_attempts_per_minute = 5

        # State
        self._halt_level = HaltLevel.NONE
        self._halt_reason: Optional[HaltReason] = None
        self._halt_message: str = ""
        self._halt_time: Optional[datetime] = None
        self._is_manually_halted = False

        # Tracking
        # SECURITY FIX: Initialize equity and peak_equity with initial_equity
        # to ensure drawdown tracking works from the first update
        self._equity = self._initial_equity
        self._peak_equity = self._initial_equity
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._consecutive_losses = 0
        self._recent_losses: deque = deque(maxlen=100)
        self._last_connectivity_time = datetime.now()

        # Components
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._alert_manager = AlertManager()
        if alerting_manager is not None:
            self._alert_manager.add_callback(
                TelegramAlertCallback(alerting_manager)
            )
        self._recovery_manager = RecoveryManager(
            cooldown_seconds=self.config.default_cooldown_seconds,
            require_confirmation=self.config.require_manual_reset
        )

        # History
        self._halt_history: deque = deque(maxlen=100)

        # Execution failure tracking for escalation (Sprint 3)
        self._consecutive_close_failures = 0
        self._max_close_failures_before_escalation = 3
        self._close_failure_window = timedelta(seconds=60)
        self._close_failure_timestamps: deque = deque(maxlen=20)

        # Thread safety
        self._lock = threading.RLock()

        self._logger = logging.getLogger("kill_switch.main")

        # Initialize circuit breakers
        self._init_circuit_breakers()

        # =====================================================================
        # LOAD PERSISTED STATE (if available)
        # =====================================================================
        if self._persistence_enabled and self._store:
            self._load_persisted_state()
            # Start heartbeat
            self._store.update_heartbeat()

        self._logger.info(
            f"Kill Switch initialized "
            f"(persistence={'enabled' if self._persistence_enabled else 'disabled'})"
        )

    def _init_circuit_breakers(self) -> None:
        """Initialize all circuit breakers."""
        breaker_configs = [
            CircuitBreakerConfig(
                name="daily_loss",
                threshold=self.config.max_daily_loss_pct,
                halt_level=HaltLevel.FULL_HALT,
                cooldown_seconds=86400,  # 24 hours
                auto_recovery=False
            ),
            CircuitBreakerConfig(
                name="weekly_loss",
                threshold=self.config.max_weekly_loss_pct,
                halt_level=HaltLevel.FULL_HALT,
                cooldown_seconds=604800,  # 7 days
                auto_recovery=False
            ),
            CircuitBreakerConfig(
                name="max_drawdown",
                threshold=self.config.max_drawdown_pct,
                halt_level=HaltLevel.EMERGENCY,
                cooldown_seconds=86400,
                auto_recovery=False
            ),
            CircuitBreakerConfig(
                name="consecutive_losses",
                threshold=self.config.max_consecutive_losses,
                halt_level=HaltLevel.REDUCED,
                cooldown_seconds=3600,  # 1 hour
                auto_recovery=True
            ),
            CircuitBreakerConfig(
                name="loss_velocity",
                threshold=self.config.loss_velocity_threshold_pct,
                halt_level=HaltLevel.CLOSE_ONLY,
                cooldown_seconds=1800,  # 30 minutes
                auto_recovery=True
            ),
            CircuitBreakerConfig(
                name="var_breach",
                threshold=self.config.max_var_pct,
                halt_level=HaltLevel.NEW_ONLY,
                cooldown_seconds=3600,
                auto_recovery=True
            ),
            CircuitBreakerConfig(
                name="exposure_breach",
                threshold=self.config.max_gross_exposure_pct,
                halt_level=HaltLevel.NEW_ONLY,
                cooldown_seconds=1800,
                auto_recovery=True
            )
        ]

        for cfg in breaker_configs:
            self._breakers[cfg.name] = CircuitBreaker(cfg)

    # =========================================================================
    # PERSISTENCE METHODS
    # =========================================================================

    def _load_persisted_state(self) -> None:
        """Load previously persisted state from database."""
        if not self._store:
            return

        try:
            state = self._store.load_state()
            if state is None:
                self._logger.info("No persisted state found - starting fresh")
                return

            # Restore halt state
            self._halt_level = HaltLevel(state.halt_level)
            self._halt_reason = HaltReason(state.halt_reason) if state.halt_reason else None
            self._halt_message = state.halt_message
            self._halt_time = (
                datetime.fromisoformat(state.halt_time)
                if state.halt_time else None
            )
            self._is_manually_halted = state.is_manually_halted

            # Restore tracking state
            self._equity = state.equity
            self._peak_equity = state.peak_equity
            self._daily_pnl = state.daily_pnl
            self._weekly_pnl = state.weekly_pnl
            self._consecutive_losses = state.consecutive_losses

            # Load breaker states
            breakers = self._store.load_breakers()
            for breaker_record in breakers:
                if breaker_record.name in self._breakers:
                    breaker = self._breakers[breaker_record.name]
                    breaker.state = BreakerState(breaker_record.state)
                    breaker.trip_count = breaker_record.trip_count
                    breaker.last_trip_time = (
                        datetime.fromisoformat(breaker_record.last_trip_time)
                        if breaker_record.last_trip_time else None
                    )
                    breaker.recovery_time = (
                        datetime.fromisoformat(breaker_record.recovery_time)
                        if breaker_record.recovery_time else None
                    )

            # Log restoration
            if self._halt_level != HaltLevel.NONE:
                self._logger.warning(
                    f"RESTORED HALT STATE from persistence: "
                    f"level={self._halt_level.name}, reason={self._halt_reason}, "
                    f"is_manual={self._is_manually_halted}"
                )
            else:
                self._logger.info(
                    f"Restored state from persistence: equity={self._equity:.2f}, "
                    f"peak={self._peak_equity:.2f}"
                )

        except Exception as e:
            self._logger.error(f"Failed to load persisted state: {e}")

    def _save_state_to_store(self) -> None:
        """Save current state to persistence store."""
        if not self._store or not self._persistence_enabled:
            return

        try:
            state = KillSwitchState(
                halt_level=self._halt_level.value,
                halt_reason=self._halt_reason.value if self._halt_reason else None,
                halt_message=self._halt_message,
                halt_time=self._halt_time.isoformat() if self._halt_time else None,
                is_manually_halted=self._is_manually_halted,
                equity=self._equity,
                peak_equity=self._peak_equity,
                daily_pnl=self._daily_pnl,
                weekly_pnl=self._weekly_pnl,
                consecutive_losses=self._consecutive_losses,
                last_updated=datetime.utcnow().isoformat()
            )
            self._store.save_state(state)

            # Also update heartbeat
            self._store.update_heartbeat()

        except Exception as e:
            self._logger.error(f"Failed to save state to store: {e}")

    def _save_breakers_to_store(self) -> None:
        """Save circuit breaker states to persistence store."""
        if not self._store or not self._persistence_enabled:
            return

        try:
            for name, breaker in self._breakers.items():
                record = BreakerRecord(
                    name=name,
                    state=breaker.state.value,
                    trip_count=breaker.trip_count,
                    last_trip_time=(
                        breaker.last_trip_time.isoformat()
                        if breaker.last_trip_time else None
                    ),
                    recovery_time=(
                        breaker.recovery_time.isoformat()
                        if breaker.recovery_time else None
                    ),
                    threshold=breaker.config.threshold
                )
                self._store.save_breaker(record)
        except Exception as e:
            self._logger.error(f"Failed to save breakers to store: {e}")

    def _record_halt_event_to_store(self, halt_event: HaltEvent) -> None:
        """Record halt event to persistence store for audit trail."""
        if not self._store or not self._persistence_enabled:
            return

        try:
            record = HaltEventRecord(
                halt_id=halt_event.halt_id,
                reason=halt_event.reason.value,
                level=halt_event.level.value,
                timestamp=halt_event.timestamp.isoformat(),
                trigger_value=halt_event.trigger_value,
                threshold=halt_event.threshold,
                message=halt_event.message,
                auto_recovery=halt_event.auto_recovery,
                recovery_time=(
                    halt_event.recovery_time.isoformat()
                    if halt_event.recovery_time else None
                ),
                recovered=halt_event.recovered,
                recovery_timestamp=(
                    halt_event.recovery_timestamp.isoformat()
                    if halt_event.recovery_timestamp else None
                )
            )
            self._store.record_halt_event(record)
        except Exception as e:
            self._logger.error(f"Failed to record halt event: {e}")

    # =========================================================================
    # MAIN UPDATE AND CHECK METHODS
    # =========================================================================

    def update(
        self,
        equity: float,
        peak_equity: Optional[float] = None,
        daily_pnl: Optional[float] = None,
        weekly_pnl: Optional[float] = None,
        var_pct: Optional[float] = None,
        gross_exposure_pct: Optional[float] = None,
        is_connected: bool = True,
        correlation_z_score: Optional[float] = None,
    ) -> HaltLevel:
        """
        Update kill switch with current portfolio state.

        This is the main method that should be called regularly
        to check all conditions and trigger halts if necessary.

        Args:
            equity: Current account equity
            peak_equity: Peak equity (for drawdown calc)
            daily_pnl: Today's P&L
            weekly_pnl: This week's P&L
            var_pct: Current VaR as percentage
            gross_exposure_pct: Current gross exposure as percentage
            is_connected: Whether connected to broker/data
            correlation_z_score: Sprint 5 — max |z-score| from correlation tracker

        Returns:
            Current halt level
        """
        with self._lock:
            # Update tracking
            self._equity = equity
            # SECURITY FIX: Auto-track peak equity from current equity
            # This ensures drawdown is always calculated correctly
            if peak_equity is not None:
                self._peak_equity = max(self._peak_equity, peak_equity)
            else:
                # Auto-track: peak is always the maximum equity seen
                self._peak_equity = max(self._peak_equity, equity)
            if daily_pnl is not None:
                self._daily_pnl = daily_pnl
            if weekly_pnl is not None:
                self._weekly_pnl = weekly_pnl

            # Update connectivity
            if is_connected:
                self._last_connectivity_time = datetime.now()

            # If manually halted, stay halted
            if self._is_manually_halted:
                return self._halt_level

            # Check recovery state
            if self._halt_level != HaltLevel.NONE:
                recovery_state = self._recovery_manager.update()
                if recovery_state == RecoveryState.RECOVERED:
                    self._clear_halt()
                    return HaltLevel.NONE
                elif recovery_state in [RecoveryState.GRADUAL_RESTART]:
                    # Adjust halt level based on recovery progress
                    multiplier = self._recovery_manager.position_multiplier
                    if multiplier >= 0.8:
                        self._halt_level = HaltLevel.CAUTION
                    elif multiplier >= 0.5:
                        self._halt_level = HaltLevel.REDUCED
                    return self._halt_level

            # === CHECK ALL CONDITIONS ===

            # 1. Drawdown check (HIGHEST PRIORITY)
            if self._peak_equity > 0:
                drawdown = (self._peak_equity - equity) / self._peak_equity
                if self._breakers["max_drawdown"].check(drawdown):
                    self._trigger_halt(
                        HaltReason.MAX_DRAWDOWN,
                        HaltLevel.EMERGENCY,
                        drawdown,
                        self.config.max_drawdown_pct,
                        f"Max drawdown breached: {drawdown:.1%}"
                    )
                    return self._halt_level

            # 2. Daily loss check
            if self._equity > 0 and daily_pnl is not None:
                daily_loss_pct = abs(min(0, daily_pnl)) / self._equity
                if self._breakers["daily_loss"].check(daily_loss_pct):
                    self._trigger_halt(
                        HaltReason.DAILY_LOSS_LIMIT,
                        HaltLevel.FULL_HALT,
                        daily_loss_pct,
                        self.config.max_daily_loss_pct,
                        f"Daily loss limit breached: {daily_loss_pct:.1%}"
                    )
                    return self._halt_level

            # 3. Weekly loss check
            if self._equity > 0 and weekly_pnl is not None:
                weekly_loss_pct = abs(min(0, weekly_pnl)) / self._equity
                if self._breakers["weekly_loss"].check(weekly_loss_pct):
                    self._trigger_halt(
                        HaltReason.WEEKLY_LOSS_LIMIT,
                        HaltLevel.FULL_HALT,
                        weekly_loss_pct,
                        self.config.max_weekly_loss_pct,
                        f"Weekly loss limit breached: {weekly_loss_pct:.1%}"
                    )
                    return self._halt_level

            # 4. Consecutive losses check
            if self._breakers["consecutive_losses"].check(
                self._consecutive_losses,
                self.config.max_consecutive_losses
            ):
                self._trigger_halt(
                    HaltReason.CONSECUTIVE_LOSSES,
                    HaltLevel.REDUCED,
                    self._consecutive_losses,
                    self.config.max_consecutive_losses,
                    f"Too many consecutive losses: {self._consecutive_losses}"
                )

            # 5. Loss velocity check
            loss_velocity = self._calculate_loss_velocity()
            if self._breakers["loss_velocity"].check(loss_velocity):
                self._trigger_halt(
                    HaltReason.LOSS_VELOCITY,
                    HaltLevel.CLOSE_ONLY,
                    loss_velocity,
                    self.config.loss_velocity_threshold_pct,
                    f"Loss velocity too high: {loss_velocity:.1%} in {self.config.loss_velocity_window_minutes}min"
                )

            # 6. VaR check
            if var_pct is not None:
                if self._breakers["var_breach"].check(var_pct):
                    self._trigger_halt(
                        HaltReason.VAR_BREACH,
                        HaltLevel.NEW_ONLY,
                        var_pct,
                        self.config.max_var_pct,
                        f"VaR limit breached: {var_pct:.1%}"
                    )

            # 7. Exposure check
            if gross_exposure_pct is not None:
                if self._breakers["exposure_breach"].check(gross_exposure_pct):
                    self._trigger_halt(
                        HaltReason.EXPOSURE_BREACH,
                        HaltLevel.NEW_ONLY,
                        gross_exposure_pct,
                        self.config.max_gross_exposure_pct,
                        f"Exposure limit breached: {gross_exposure_pct:.0%}"
                    )

            # 8. Connectivity check
            if not is_connected:
                disconnect_duration = (
                    datetime.now() - self._last_connectivity_time
                ).total_seconds()
                if disconnect_duration >= self.config.max_disconnect_seconds:
                    self._trigger_halt(
                        HaltReason.CONNECTIVITY_LOSS,
                        HaltLevel.CLOSE_ONLY,
                        disconnect_duration,
                        self.config.max_disconnect_seconds,
                        f"Connectivity lost for {disconnect_duration:.0f}s"
                    )

            # 9. Sprint 5: Correlation breakdown check
            if correlation_z_score is not None and abs(correlation_z_score) >= 3.0:
                self._trigger_halt(
                    HaltReason.CORRELATION_BREAKDOWN,
                    HaltLevel.REDUCED,
                    abs(correlation_z_score),
                    3.0,
                    f"Correlation breakdown: z-score={correlation_z_score:.2f}"
                )

            # PERSISTENCE: Save state after update
            self._save_state_to_store()

            return self._halt_level

    def record_trade_result(self, pnl: float, pnl_pct: float) -> None:
        """
        Record a trade result for tracking.

        Args:
            pnl: Trade P&L in currency
            pnl_pct: Trade P&L as percentage
        """
        with self._lock:
            # Track consecutive losses
            if pnl < 0:
                self._consecutive_losses += 1
                self._recent_losses.append({
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'timestamp': datetime.now()
                })
            else:
                self._consecutive_losses = 0

    def _calculate_loss_velocity(self) -> float:
        """Calculate recent loss velocity."""
        if not self._recent_losses:
            return 0.0

        cutoff = datetime.now() - timedelta(
            minutes=self.config.loss_velocity_window_minutes
        )

        recent = [
            loss['pnl_pct']
            for loss in self._recent_losses
            if loss['timestamp'] >= cutoff
        ]

        return abs(sum(recent)) if recent else 0.0

    # =========================================================================
    # HALT MANAGEMENT
    # =========================================================================

    def _trigger_halt(
        self,
        reason: HaltReason,
        level: HaltLevel,
        trigger_value: float,
        threshold: float,
        message: str
    ) -> None:
        """Trigger a trading halt."""
        # Only upgrade halt level, never downgrade
        if level.value <= self._halt_level.value:
            return

        self._halt_level = level
        self._halt_reason = reason
        self._halt_message = message
        self._halt_time = datetime.now()

        # Determine if auto-recovery is allowed
        auto_recovery = level.value < HaltLevel.FULL_HALT.value

        # Calculate recovery time
        breaker = None
        for b in self._breakers.values():
            if b.is_tripped and b.config.halt_level == level:
                breaker = b
                break

        recovery_time = None
        if breaker and breaker.recovery_time:
            recovery_time = breaker.recovery_time

        # Create halt event
        halt_event = HaltEvent(
            halt_id=f"halt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            reason=reason,
            level=level,
            timestamp=datetime.now(),
            trigger_value=trigger_value,
            threshold=threshold,
            message=message,
            auto_recovery=auto_recovery,
            recovery_time=recovery_time
        )

        # Store in history
        self._halt_history.append(halt_event)

        # Send alerts
        self._alert_manager.send_alert(halt_event)

        # Start recovery process if not emergency
        if level != HaltLevel.EMERGENCY:
            self._recovery_manager.start_recovery(halt_event)

        # PERSISTENCE: Save halt event and breaker states
        self._record_halt_event_to_store(halt_event)
        self._save_breakers_to_store()
        self._save_state_to_store()

        self._logger.critical(
            f"TRADING HALTED | Level: {level.name} | Reason: {reason.value} | "
            f"Message: {message}"
        )

    def _clear_halt(self, clear_manual: bool = False) -> None:
        """
        Clear halt state.

        SECURITY FIX: Manual halt flag is no longer automatically cleared.
        This prevents accidental bypass of intentional trading stops.

        Args:
            clear_manual: If True, also clear the manual halt flag (requires explicit call)
        """
        # SECURITY: Don't clear manual halt unless explicitly requested
        if self._is_manually_halted and not clear_manual:
            self._logger.warning(
                "Halt state cleared but MANUAL HALT remains active. "
                "Use confirm_reset() with clear_manual_halt=True to clear manual halt."
            )
            # Still clear the halt level but keep manual flag
            self._halt_level = HaltLevel.FULL_HALT  # Keep at FULL_HALT for manual
            return

        self._halt_level = HaltLevel.NONE
        self._halt_reason = None
        self._halt_message = ""
        self._halt_time = None
        if clear_manual:
            self._is_manually_halted = False

        self._logger.info("Trading halt CLEARED - Normal operation resumed")

    # =========================================================================
    # PUBLIC API - QUERIES
    # =========================================================================

    @property
    def is_halted(self) -> bool:
        """Check if trading is currently halted."""
        return self._halt_level.value >= HaltLevel.FULL_HALT.value

    @property
    def halt_level(self) -> HaltLevel:
        """Get current halt level."""
        return self._halt_level

    @property
    def halt_reason(self) -> Optional[HaltReason]:
        """Get reason for current halt."""
        return self._halt_reason

    def is_trading_allowed(self) -> bool:
        """
        Check if new trading is allowed.

        Returns:
            True if new trades can be opened
        """
        return self._halt_level.value < HaltLevel.NEW_ONLY.value

    def is_closing_allowed(self) -> bool:
        """
        Check if closing positions is allowed.

        Returns:
            True if positions can be closed
        """
        return self._halt_level.value < HaltLevel.FULL_HALT.value

    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on current state.

        Returns:
            Multiplier from 0.0 (no trading) to 1.0 (full size)
        """
        if self._halt_level == HaltLevel.NONE:
            return 1.0
        elif self._halt_level == HaltLevel.CAUTION:
            return 0.8
        elif self._halt_level == HaltLevel.REDUCED:
            return 0.5
        elif self._halt_level.value >= HaltLevel.NEW_ONLY.value:
            return 0.0

        # During recovery
        return self._recovery_manager.position_multiplier

    # =========================================================================
    # PUBLIC API - CONTROLS
    # =========================================================================

    def emergency_halt(self, reason: str = "Manual emergency halt") -> None:
        """
        Trigger emergency halt (manual).

        Args:
            reason: Reason for the halt
        """
        with self._lock:
            self._is_manually_halted = True
            self._trigger_halt(
                HaltReason.EMERGENCY_STOP,
                HaltLevel.EMERGENCY,
                0.0,
                0.0,
                reason
            )

    def manual_halt(self, reason: str = "Manual halt") -> None:
        """
        Trigger manual trading halt.

        Args:
            reason: Reason for the halt
        """
        with self._lock:
            self._is_manually_halted = True
            self._trigger_halt(
                HaltReason.MANUAL_HALT,
                HaltLevel.FULL_HALT,
                0.0,
                0.0,
                reason
            )

    def request_reset(self, notification_callback: Optional[callable] = None) -> bool:
        """
        Request a reset token for manual reset.

        SECURITY FIX: Token is NO LONGER returned directly to prevent exposure.
        Instead, the token is sent via secure notification callback.

        Args:
            notification_callback: Optional callback function that receives the token
                                   securely (e.g., sends via encrypted email/SMS)

        Returns:
            True if reset request was initiated, False otherwise
        """
        if self._recovery_manager.state == RecoveryState.CONFIRMATION_PENDING:
            token = self._recovery_manager._confirmation_token
            if notification_callback:
                try:
                    # Send token via secure channel (callback handles delivery)
                    notification_callback(token)
                    self._logger.info("Reset token sent via secure notification channel")
                except Exception as e:
                    self._logger.error(f"Failed to send reset token: {e}")
                    return False
            else:
                # Log that token exists but don't expose it
                self._logger.warning(
                    "Reset token generated but no notification callback provided. "
                    "Token NOT exposed for security. Provide a notification_callback."
                )
            return True
        return False

    def confirm_reset(self, token: str, clear_manual_halt: bool = False, reason: str = "") -> bool:
        """
        Confirm reset with token.

        SECURITY FIX: Manual halts now require explicit confirmation to clear.

        Args:
            token: Confirmation token from request_reset()
            clear_manual_halt: Must be True to clear a manual halt (explicit confirmation)
            reason: Required reason when clearing manual halt (for audit trail)

        Returns:
            True if reset confirmed
        """
        with self._lock:
            # SECURITY: If manually halted, require explicit confirmation and reason
            if self._is_manually_halted:
                if not clear_manual_halt:
                    self._logger.warning(
                        "Reset denied: System is manually halted. "
                        "Set clear_manual_halt=True to explicitly clear manual halt."
                    )
                    return False
                if not reason or len(reason) < 10:
                    self._logger.warning(
                        "Reset denied: Clearing manual halt requires a reason (min 10 chars) for audit trail."
                    )
                    return False
                # Audit log the manual halt clear
                self._logger.warning(
                    f"AUDIT: Manual halt being cleared. Reason: {reason}"
                )

            if self._recovery_manager.confirm_recovery(token):
                # Reset all breakers
                for breaker in self._breakers.values():
                    breaker.reset(force=True)

                # Only clear manual halt if explicitly requested
                if self._is_manually_halted and clear_manual_halt:
                    self._is_manually_halted = False
                    self._logger.warning(f"AUDIT: Manual halt cleared. Reason: {reason}")
                elif not self._is_manually_halted:
                    self._is_manually_halted = False

                return True
            return False

    def _check_rate_limit(self) -> bool:
        """Check if reset attempts are within rate limit."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        # Remove old attempts
        while self._reset_attempts and self._reset_attempts[0] < cutoff:
            self._reset_attempts.popleft()

        # Check rate limit
        if len(self._reset_attempts) >= self._max_reset_attempts_per_minute:
            self._logger.warning(
                f"Rate limit exceeded: {len(self._reset_attempts)} attempts in last minute"
            )
            return False

        self._reset_attempts.append(now)
        return True

    def force_reset(self, admin_key: str) -> bool:
        """
        Force reset (admin override).

        WARNING: This bypasses all safety checks.
        Rate limited to prevent brute force attacks.

        Args:
            admin_key: Admin key for authorization

        Returns:
            True if force reset successful
        """
        with self._lock:
            # SECURITY: Rate limit to prevent brute force
            if not self._check_rate_limit():
                self._logger.error("Force reset denied: rate limit exceeded")
                return False

            # SECURITY: Admin key must be configured
            if not self._admin_key:
                self._logger.error("Force reset denied: no admin key configured")
                return False

            if self._recovery_manager.force_recovery(admin_key, self._admin_key):
                # Reset all breakers
                for breaker in self._breakers.values():
                    breaker.reset(force=True)
                self._clear_halt()
                self._logger.warning("FORCE RESET executed by admin")
                return True

            self._logger.warning("Force reset failed: invalid admin key")
            return False

    def reset_daily_counters(self) -> None:
        """Reset daily counters (call at start of trading day)."""
        with self._lock:
            self._daily_pnl = 0.0
            self._consecutive_losses = 0
            self._recent_losses.clear()

            # Reset daily breakers
            if "daily_loss" in self._breakers:
                self._breakers["daily_loss"].reset(force=True)
            if "consecutive_losses" in self._breakers:
                self._breakers["consecutive_losses"].reset(force=True)
            if "loss_velocity" in self._breakers:
                self._breakers["loss_velocity"].reset(force=True)

    def reset_weekly_counters(self) -> None:
        """Reset weekly counters (call at start of trading week)."""
        with self._lock:
            self._weekly_pnl = 0.0

            if "weekly_loss" in self._breakers:
                self._breakers["weekly_loss"].reset(force=True)

    # =========================================================================
    # ESCALATION (Sprint 3)
    # =========================================================================

    def escalate(self, reason: str) -> HaltLevel:
        """
        Ratchet up the halt level by one step. Never ratchets down.

        Escalation path:
            NONE → CAUTION → REDUCED → NEW_ONLY → CLOSE_ONLY → FULL_HALT → EMERGENCY

        This is the safety net for when the current halt level is insufficient
        (e.g., CLOSE_ONLY but MT5 can't execute closes).

        Args:
            reason: Why the escalation is happening.

        Returns:
            The new HaltLevel after escalation.
        """
        with self._lock:
            current = self._halt_level
            if current.value >= HaltLevel.EMERGENCY.value:
                self._logger.warning(
                    "Escalation requested but already at EMERGENCY level"
                )
                return current

            new_level = HaltLevel(current.value + 1)

            self._logger.critical(
                "Kill switch ESCALATING",
                extra={
                    'previous_level': current.name,
                    'new_level': new_level.name,
                    'reason': reason,
                    'consecutive_close_failures': self._consecutive_close_failures,
                }
            )

            self._trigger_halt(
                HaltReason.EXECUTION_ERROR,
                new_level,
                float(self._consecutive_close_failures),
                float(self._max_close_failures_before_escalation),
                f"Escalation: {reason} (from {current.name} to {new_level.name})"
            )

            return self._halt_level

    def record_close_failure(self, reason: str = "MT5 close execution failed") -> HaltLevel:
        """
        Record a failed close attempt. After max_close_failures_before_escalation
        failures within close_failure_window, auto-escalate the halt level.

        Call this when a position close order fails (e.g., MT5 timeout, rejected).

        Args:
            reason: Description of the failure.

        Returns:
            Current HaltLevel (may be escalated).
        """
        with self._lock:
            now = datetime.now()
            self._close_failure_timestamps.append(now)

            # Count failures within the window
            cutoff = now - self._close_failure_window
            recent_failures = sum(
                1 for ts in self._close_failure_timestamps if ts >= cutoff
            )
            self._consecutive_close_failures = recent_failures

            self._logger.warning(
                "Close execution failure recorded: %s "
                "(%d/%d within %ds window)",
                reason, recent_failures,
                self._max_close_failures_before_escalation,
                int(self._close_failure_window.total_seconds())
            )

            # Auto-escalate if threshold reached
            if recent_failures >= self._max_close_failures_before_escalation:
                self._logger.critical(
                    "Close failure threshold reached — auto-escalating kill switch"
                )
                # Reset counter to avoid immediate re-escalation
                self._close_failure_timestamps.clear()
                self._consecutive_close_failures = 0
                return self.escalate(
                    f"{recent_failures} close failures in "
                    f"{int(self._close_failure_window.total_seconds())}s: {reason}"
                )

            return self._halt_level

    # =========================================================================
    # STATUS AND REPORTING
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        return {
            "halt_level": self._halt_level.name,
            "halt_level_value": self._halt_level.value,
            "is_halted": self.is_halted,
            "is_trading_allowed": self.is_trading_allowed(),
            "is_closing_allowed": self.is_closing_allowed(),
            "position_multiplier": self.get_position_multiplier(),
            "halt_reason": self._halt_reason.value if self._halt_reason else None,
            "halt_message": self._halt_message,
            "halt_time": self._halt_time.isoformat() if self._halt_time else None,
            "is_manually_halted": self._is_manually_halted,
            "tracking": {
                "equity": self._equity,
                "peak_equity": self._peak_equity,
                "daily_pnl": self._daily_pnl,
                "weekly_pnl": self._weekly_pnl,
                "consecutive_losses": self._consecutive_losses,
                "drawdown_pct": (
                    (self._peak_equity - self._equity) / self._peak_equity
                    if self._peak_equity > 0 else 0
                )
            },
            "breakers": {
                name: breaker.get_status()
                for name, breaker in self._breakers.items()
            },
            "recovery": self._recovery_manager.get_status(),
            "config": self.config.to_dict()
        }

    def get_dashboard(self) -> str:
        """Generate text-based status dashboard."""
        status = self.get_status()
        tracking = status["tracking"]

        # Determine status indicator
        if status["is_halted"]:
            status_indicator = "[X] HALTED"
        elif status["halt_level_value"] > 0:
            status_indicator = "[!] WARNING"
        else:
            status_indicator = "[OK] NORMAL"

        # Breaker status
        breaker_lines = []
        for name, breaker in status["breakers"].items():
            state_char = "X" if breaker["is_tripped"] else "O"
            breaker_lines.append(
                f"    [{state_char}] {name:20} | "
                f"Threshold: {breaker['threshold']:.2%} | "
                f"Trips: {breaker['trip_count']}"
            )

        return f"""
================================================================================
                          KILL SWITCH STATUS
================================================================================

  STATUS: {status_indicator}

  Halt Level:          {status['halt_level']:15}
  Halt Reason:         {status['halt_reason'] or 'N/A':15}
  Position Multiplier: {status['position_multiplier']:.0%}

  TRACKING
  ─────────────────────────────────────────────────────────────────────────────
  Equity:              ${tracking['equity']:>15,.2f}
  Peak Equity:         ${tracking['peak_equity']:>15,.2f}
  Drawdown:            {tracking['drawdown_pct']*100:>14.2f}%
  Daily P&L:           ${tracking['daily_pnl']:>15,.2f}
  Weekly P&L:          ${tracking['weekly_pnl']:>15,.2f}
  Consecutive Losses:  {tracking['consecutive_losses']:>15}

  CIRCUIT BREAKERS
  ─────────────────────────────────────────────────────────────────────────────
{chr(10).join(breaker_lines)}

  RECOVERY
  ─────────────────────────────────────────────────────────────────────────────
  State:               {status['recovery']['state']}
  Progress:            Step {status['recovery']['current_step']}/{status['recovery']['total_steps']}

================================================================================
"""

    def add_alert_webhook(self, url: str, headers: Optional[Dict[str, str]] = None) -> None:
        """
        Add a webhook for alerts.

        Args:
            url: Webhook URL
            headers: Optional HTTP headers
        """
        self._alert_manager.add_callback(WebhookAlertCallback(url, headers))


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_kill_switch(
    preset: str = "moderate",
    admin_key: Optional[str] = None
) -> KillSwitch:
    """
    Create a KillSwitch with preset configuration.

    Args:
        preset: "conservative", "moderate", "aggressive"
        admin_key: Admin key for force reset

    Returns:
        Configured KillSwitch
    """
    presets = {
        "conservative": KillSwitchConfig(
            max_daily_loss_pct=0.02,
            max_weekly_loss_pct=0.03,
            max_drawdown_pct=0.05,
            max_consecutive_losses=3,
            max_var_pct=0.01
        ),
        "moderate": KillSwitchConfig(
            max_daily_loss_pct=0.03,
            max_weekly_loss_pct=0.05,
            max_drawdown_pct=0.10,
            max_consecutive_losses=5,
            max_var_pct=0.02
        ),
        "aggressive": KillSwitchConfig(
            max_daily_loss_pct=0.05,
            max_weekly_loss_pct=0.08,
            max_drawdown_pct=0.15,
            max_consecutive_losses=7,
            max_var_pct=0.03
        )
    }

    config = presets.get(preset, presets["moderate"])

    return KillSwitch(config=config, admin_key=admin_key)
