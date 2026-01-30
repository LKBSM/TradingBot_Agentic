# =============================================================================
# LIVE RISK MANAGER - Real-time Risk Enforcement
# =============================================================================
# Production risk management for live trading.
#
# Features:
#   - Real-time drawdown monitoring
#   - Position size validation
#   - Kill switch integration
#   - Daily/Weekly loss limits
#   - Leverage monitoring
#   - Risk metrics calculation
#
# =============================================================================

import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, Thread
import numpy as np

from .mt5_connector import MT5Connector, AccountInfo, PositionInfo
from .alerting import AlertManager, AlertLevel


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LiveRiskConfig:
    """Configuration for live risk management."""

    # Drawdown limits
    max_drawdown_pct: float = 10.0           # 10% max account drawdown
    daily_loss_limit_pct: float = 3.0        # 3% max daily loss
    weekly_loss_limit_pct: float = 6.0       # 6% max weekly loss

    # Position limits
    max_position_size_pct: float = 20.0      # 20% max position value
    max_risk_per_trade_pct: float = 1.0      # 1% max risk per trade
    max_leverage: float = 1.0                 # No leverage by default

    # Trade limits
    max_trades_per_day: int = 10
    max_consecutive_losses: int = 4
    cooldown_after_loss_minutes: int = 15

    # Kill switch
    kill_switch_enabled: bool = True
    kill_switch_dd_threshold: float = 8.0    # Kill switch at 8% DD (before max)
    auto_close_on_kill: bool = False         # Close positions on kill switch

    # Monitoring
    check_interval_seconds: int = 5
    equity_update_interval: int = 1


# =============================================================================
# RISK METRICS
# =============================================================================

@dataclass
class RiskMetrics:
    """Current risk metrics snapshot."""
    timestamp: datetime = field(default_factory=datetime.now)

    # Account
    balance: float = 0.0
    equity: float = 0.0
    margin_used: float = 0.0
    free_margin: float = 0.0

    # Drawdown
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0

    # Daily/Weekly
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    weekly_pnl: float = 0.0
    weekly_pnl_pct: float = 0.0

    # Position
    position_count: int = 0
    total_exposure: float = 0.0
    current_leverage: float = 0.0

    # Trade stats
    trades_today: int = 0
    consecutive_losses: int = 0

    # Risk status
    is_kill_switch_active: bool = False
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'balance': self.balance,
            'equity': self.equity,
            'margin_used': self.margin_used,
            'free_margin': self.free_margin,
            'peak_equity': self.peak_equity,
            'current_drawdown_pct': self.current_drawdown_pct,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl_pct,
            'weekly_pnl': self.weekly_pnl,
            'weekly_pnl_pct': self.weekly_pnl_pct,
            'position_count': self.position_count,
            'total_exposure': self.total_exposure,
            'current_leverage': self.current_leverage,
            'trades_today': self.trades_today,
            'consecutive_losses': self.consecutive_losses,
            'is_kill_switch_active': self.is_kill_switch_active,
            'risk_level': self.risk_level,
        }


# =============================================================================
# LIVE RISK MANAGER
# =============================================================================

class LiveRiskManager:
    """
    Real-time risk management for live trading.

    Responsibilities:
    1. Monitor account equity and drawdown
    2. Enforce position size limits
    3. Track daily/weekly P&L
    4. Activate kill switch when thresholds breached
    5. Validate trades before execution
    6. Calculate risk metrics

    Usage:
        risk_mgr = LiveRiskManager(connector, config, alerts)
        risk_mgr.start()

        # Check if trade is allowed
        if risk_mgr.can_trade():
            # Validate specific trade
            allowed, reason = risk_mgr.validate_trade(
                direction="BUY",
                volume=0.1,
                sl_distance=10.0
            )
            if allowed:
                # Execute trade
                pass
    """

    def __init__(
        self,
        connector: MT5Connector,
        config: LiveRiskConfig = None,
        alert_manager: AlertManager = None
    ):
        """
        Initialize LiveRiskManager.

        Args:
            connector: MT5Connector instance
            config: LiveRiskConfig
            alert_manager: AlertManager for notifications
        """
        self.connector = connector
        self.config = config or LiveRiskConfig()
        self.alerts = alert_manager

        self._logger = logging.getLogger("LiveRiskManager")
        self._lock = Lock()

        # State tracking
        self._peak_equity = 0.0
        self._daily_start_equity = 0.0
        self._weekly_start_equity = 0.0
        self._last_daily_reset = datetime.now().date()
        self._last_weekly_reset = datetime.now().date()

        # Trade tracking
        self._trades_today = 0
        self._consecutive_losses = 0
        self._last_loss_time: Optional[datetime] = None

        # Kill switch
        self._kill_switch_active = False
        self._kill_switch_reason = ""
        self._kill_switch_time: Optional[datetime] = None

        # Metrics history
        self._metrics_history: List[RiskMetrics] = []
        self._current_metrics = RiskMetrics()

        # Monitoring thread
        self._running = False
        self._monitor_thread: Optional[Thread] = None

        # Callbacks
        self._on_kill_switch: List[Callable] = []
        self._on_drawdown_warning: List[Callable] = []

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self):
        """Start risk monitoring."""
        if self._running:
            return

        # Initialize equity tracking
        account = self.connector.get_account_info()
        self._peak_equity = account.equity
        self._daily_start_equity = account.equity
        self._weekly_start_equity = account.equity

        self._running = True
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        self._logger.info(
            f"LiveRiskManager started: equity=${account.equity:.2f}, "
            f"max_dd={self.config.max_drawdown_pct}%"
        )

    def stop(self):
        """Stop risk monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self._logger.info("LiveRiskManager stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        last_check = 0

        while self._running:
            try:
                now = time.time()

                # Update metrics at configured interval
                if now - last_check >= self.config.check_interval_seconds:
                    self._update_metrics()
                    self._check_risk_limits()
                    self._check_date_resets()
                    last_check = now

                time.sleep(1)

            except Exception as e:
                self._logger.error(f"Monitor error: {e}")
                time.sleep(5)

    # =========================================================================
    # METRICS UPDATE
    # =========================================================================

    def _update_metrics(self):
        """Update current risk metrics."""
        try:
            account = self.connector.get_account_info()
            positions = self.connector.get_positions()

            with self._lock:
                # Update peak equity
                if account.equity > self._peak_equity:
                    self._peak_equity = account.equity

                # Calculate drawdown
                if self._peak_equity > 0:
                    drawdown_pct = (self._peak_equity - account.equity) / self._peak_equity * 100
                else:
                    drawdown_pct = 0.0

                # Calculate daily P&L
                daily_pnl = account.equity - self._daily_start_equity
                daily_pnl_pct = (daily_pnl / self._daily_start_equity * 100) if self._daily_start_equity > 0 else 0.0

                # Calculate weekly P&L
                weekly_pnl = account.equity - self._weekly_start_equity
                weekly_pnl_pct = (weekly_pnl / self._weekly_start_equity * 100) if self._weekly_start_equity > 0 else 0.0

                # Calculate exposure
                total_exposure = sum(p.volume * p.price_current for p in positions)
                current_leverage = total_exposure / account.equity if account.equity > 0 else 0.0

                # Determine risk level
                risk_level = self._calculate_risk_level(drawdown_pct, daily_pnl_pct)

                # Update metrics
                self._current_metrics = RiskMetrics(
                    timestamp=datetime.now(),
                    balance=account.balance,
                    equity=account.equity,
                    margin_used=account.margin,
                    free_margin=account.free_margin,
                    peak_equity=self._peak_equity,
                    current_drawdown_pct=drawdown_pct,
                    daily_pnl=daily_pnl,
                    daily_pnl_pct=daily_pnl_pct,
                    weekly_pnl=weekly_pnl,
                    weekly_pnl_pct=weekly_pnl_pct,
                    position_count=len(positions),
                    total_exposure=total_exposure,
                    current_leverage=current_leverage,
                    trades_today=self._trades_today,
                    consecutive_losses=self._consecutive_losses,
                    is_kill_switch_active=self._kill_switch_active,
                    risk_level=risk_level,
                )

                # Store in history (keep last hour)
                self._metrics_history.append(self._current_metrics)
                cutoff = datetime.now() - timedelta(hours=1)
                self._metrics_history = [m for m in self._metrics_history if m.timestamp > cutoff]

        except Exception as e:
            self._logger.error(f"Metrics update error: {e}")

    def _calculate_risk_level(self, drawdown_pct: float, daily_pnl_pct: float) -> str:
        """Calculate overall risk level."""
        dd_ratio = drawdown_pct / self.config.max_drawdown_pct
        daily_ratio = abs(daily_pnl_pct) / self.config.daily_loss_limit_pct if daily_pnl_pct < 0 else 0

        max_ratio = max(dd_ratio, daily_ratio)

        if max_ratio >= 0.9:
            return "CRITICAL"
        elif max_ratio >= 0.7:
            return "HIGH"
        elif max_ratio >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    # =========================================================================
    # RISK CHECKS
    # =========================================================================

    def _check_risk_limits(self):
        """Check if any risk limits are breached."""
        metrics = self._current_metrics

        # Check kill switch threshold
        if (self.config.kill_switch_enabled and
            not self._kill_switch_active and
            metrics.current_drawdown_pct >= self.config.kill_switch_dd_threshold):
            self._activate_kill_switch(
                f"Drawdown threshold reached: {metrics.current_drawdown_pct:.1f}%"
            )
            return

        # Check max drawdown
        if metrics.current_drawdown_pct >= self.config.max_drawdown_pct:
            self._activate_kill_switch(
                f"Max drawdown exceeded: {metrics.current_drawdown_pct:.1f}% >= {self.config.max_drawdown_pct}%"
            )
            return

        # Check daily loss limit
        if metrics.daily_pnl_pct <= -self.config.daily_loss_limit_pct:
            self._activate_kill_switch(
                f"Daily loss limit exceeded: {metrics.daily_pnl_pct:.1f}%"
            )
            return

        # Check weekly loss limit
        if metrics.weekly_pnl_pct <= -self.config.weekly_loss_limit_pct:
            self._activate_kill_switch(
                f"Weekly loss limit exceeded: {metrics.weekly_pnl_pct:.1f}%"
            )
            return

        # Drawdown warning (at 70% of limit)
        warning_threshold = self.config.max_drawdown_pct * 0.7
        if metrics.current_drawdown_pct >= warning_threshold:
            if self.alerts:
                self.alerts.drawdown_warning(
                    metrics.current_drawdown_pct / 100,
                    self.config.max_drawdown_pct / 100
                )
            for callback in self._on_drawdown_warning:
                callback(metrics)

    def _check_date_resets(self):
        """Check and reset daily/weekly counters."""
        today = datetime.now().date()

        # Daily reset
        if today != self._last_daily_reset:
            self._logger.info(f"Daily reset: P&L was ${self._current_metrics.daily_pnl:.2f}")
            self._daily_start_equity = self._current_metrics.equity
            self._trades_today = 0
            self._last_daily_reset = today

        # Weekly reset (Monday)
        if today.weekday() == 0 and today != self._last_weekly_reset:
            self._logger.info(f"Weekly reset: P&L was ${self._current_metrics.weekly_pnl:.2f}")
            self._weekly_start_equity = self._current_metrics.equity
            self._last_weekly_reset = today

    # =========================================================================
    # KILL SWITCH
    # =========================================================================

    def _activate_kill_switch(self, reason: str):
        """Activate the kill switch."""
        with self._lock:
            if self._kill_switch_active:
                return

            self._kill_switch_active = True
            self._kill_switch_reason = reason
            self._kill_switch_time = datetime.now()

        self._logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

        # Send alert
        if self.alerts:
            self.alerts.kill_switch_triggered(
                reason=reason,
                equity=self._current_metrics.equity
            )

        # Execute callbacks
        for callback in self._on_kill_switch:
            try:
                callback(reason)
            except Exception as e:
                self._logger.error(f"Kill switch callback error: {e}")

        # Auto-close positions if configured
        if self.config.auto_close_on_kill:
            self._close_all_positions()

    def _close_all_positions(self):
        """Emergency close all positions."""
        try:
            results = self.connector.close_all_positions()
            closed = sum(1 for r in results if r.success)
            self._logger.warning(f"Emergency close: {closed}/{len(results)} positions closed")
        except Exception as e:
            self._logger.error(f"Emergency close failed: {e}")

    def deactivate_kill_switch(self, confirm: bool = False):
        """
        Manually deactivate the kill switch.

        Args:
            confirm: Must be True to confirm deactivation
        """
        if not confirm:
            self._logger.warning("Kill switch deactivation requires confirm=True")
            return False

        with self._lock:
            self._kill_switch_active = False
            self._kill_switch_reason = ""
            self._kill_switch_time = None

        self._logger.info("Kill switch deactivated manually")
        return True

    # =========================================================================
    # TRADE VALIDATION
    # =========================================================================

    def can_trade(self) -> bool:
        """
        Check if trading is currently allowed.

        Returns:
            True if trading is allowed
        """
        if self._kill_switch_active:
            return False

        # Check trade limits
        if self._trades_today >= self.config.max_trades_per_day:
            return False

        # Check consecutive losses
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            # Check if cooldown has passed
            if self._last_loss_time:
                cooldown = timedelta(minutes=self.config.cooldown_after_loss_minutes)
                if datetime.now() - self._last_loss_time < cooldown:
                    return False

        return True

    def validate_trade(
        self,
        direction: str,
        volume: float,
        sl_distance: float,
        entry_price: float = None
    ) -> tuple[bool, str]:
        """
        Validate a proposed trade.

        Args:
            direction: "BUY" or "SELL"
            volume: Position size in lots
            sl_distance: Stop loss distance in price units
            entry_price: Entry price (optional, uses current if not provided)

        Returns:
            Tuple of (is_allowed, reason)
        """
        if not self.can_trade():
            if self._kill_switch_active:
                return False, f"Kill switch active: {self._kill_switch_reason}"
            return False, "Trading not allowed (limits reached)"

        metrics = self._current_metrics

        # Get entry price if not provided
        if entry_price is None:
            try:
                bid, ask = self.connector.get_current_price("XAUUSD")
                entry_price = ask if direction == "BUY" else bid
            except Exception:
                return False, "Could not get current price"

        # Calculate position value and risk
        position_value = volume * entry_price
        risk_amount = volume * sl_distance

        # Check position size limit
        position_pct = (position_value / metrics.equity) * 100
        if position_pct > self.config.max_position_size_pct:
            return False, f"Position size {position_pct:.1f}% exceeds limit {self.config.max_position_size_pct}%"

        # Check risk per trade limit
        risk_pct = (risk_amount / metrics.equity) * 100
        if risk_pct > self.config.max_risk_per_trade_pct:
            return False, f"Risk per trade {risk_pct:.2f}% exceeds limit {self.config.max_risk_per_trade_pct}%"

        # Check leverage
        total_exposure = metrics.total_exposure + position_value
        proposed_leverage = total_exposure / metrics.equity
        if proposed_leverage > self.config.max_leverage:
            return False, f"Leverage {proposed_leverage:.2f}x exceeds limit {self.config.max_leverage}x"

        return True, "Trade validated"

    def calculate_position_size(
        self,
        sl_distance: float,
        entry_price: float,
        risk_pct: float = None
    ) -> float:
        """
        Calculate optimal position size based on risk.

        Args:
            sl_distance: Stop loss distance in price units
            entry_price: Entry price
            risk_pct: Risk percentage (uses config if not provided)

        Returns:
            Position size in lots
        """
        risk_pct = risk_pct or self.config.max_risk_per_trade_pct
        equity = self._current_metrics.equity

        # Risk amount in dollars
        risk_amount = equity * (risk_pct / 100)

        # Position size = Risk Amount / SL Distance
        position_size = risk_amount / sl_distance

        # Get symbol info for normalization
        try:
            symbol_info = self.connector.get_symbol_info("XAUUSD")
            # Round to volume step
            step = symbol_info.volume_step
            position_size = round(position_size / step) * step
            # Clamp to limits
            position_size = max(symbol_info.volume_min, min(symbol_info.volume_max, position_size))
        except Exception:
            pass

        return round(position_size, 2)

    # =========================================================================
    # TRADE TRACKING
    # =========================================================================

    def record_trade(self, is_winner: bool, pnl: float = 0.0):
        """
        Record a completed trade.

        Args:
            is_winner: True if trade was profitable
            pnl: Trade P&L in dollars
        """
        with self._lock:
            self._trades_today += 1

            if is_winner:
                self._consecutive_losses = 0
            else:
                self._consecutive_losses += 1
                self._last_loss_time = datetime.now()

                # Check if we need to enter cooldown
                if self._consecutive_losses >= self.config.max_consecutive_losses:
                    self._logger.warning(
                        f"Consecutive loss limit reached: {self._consecutive_losses} losses"
                    )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        return self._current_metrics

    def get_metrics_history(self) -> List[RiskMetrics]:
        """Get metrics history (last hour)."""
        return self._metrics_history.copy()

    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return self._kill_switch_active

    def get_kill_switch_reason(self) -> str:
        """Get kill switch activation reason."""
        return self._kill_switch_reason

    def register_kill_switch_callback(self, callback: Callable[[str], None]):
        """Register callback for kill switch activation."""
        self._on_kill_switch.append(callback)

    def register_drawdown_warning_callback(self, callback: Callable[[RiskMetrics], None]):
        """Register callback for drawdown warnings."""
        self._on_drawdown_warning.append(callback)

    def get_dashboard(self) -> str:
        """Get text-based risk dashboard."""
        m = self._current_metrics

        status = "HALTED" if self._kill_switch_active else "ACTIVE"

        return f"""
================================================================================
                         LIVE RISK DASHBOARD
================================================================================
 Status:          {status:12} | Risk Level:    {m.risk_level}
 Kill Switch:     {'ACTIVE - ' + self._kill_switch_reason[:30] if self._kill_switch_active else 'OFF'}
--------------------------------------------------------------------------------
 ACCOUNT
   Balance:       ${m.balance:>12,.2f}
   Equity:        ${m.equity:>12,.2f}
   Free Margin:   ${m.free_margin:>12,.2f}
--------------------------------------------------------------------------------
 DRAWDOWN
   Peak Equity:   ${m.peak_equity:>12,.2f}
   Current DD:    {m.current_drawdown_pct:>12.2f}%  (Limit: {self.config.max_drawdown_pct}%)
--------------------------------------------------------------------------------
 P&L
   Daily:         ${m.daily_pnl:>+12,.2f}  ({m.daily_pnl_pct:+.2f}%)
   Weekly:        ${m.weekly_pnl:>+12,.2f}  ({m.weekly_pnl_pct:+.2f}%)
--------------------------------------------------------------------------------
 EXPOSURE
   Positions:     {m.position_count:>12}
   Total Value:   ${m.total_exposure:>12,.2f}
   Leverage:      {m.current_leverage:>12.2f}x  (Limit: {self.config.max_leverage}x)
--------------------------------------------------------------------------------
 TRADING
   Trades Today:  {m.trades_today:>12}  (Limit: {self.config.max_trades_per_day})
   Consec. Losses: {m.consecutive_losses:>11}  (Limit: {self.config.max_consecutive_losses})
================================================================================
"""
