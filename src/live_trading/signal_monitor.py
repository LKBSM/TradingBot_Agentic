"""
Signal Monitor — Background service for operational alerting.

Provides two functions:
1. No-signal detection: alerts if no signal generated for 2 hours during market hours.
2. Daily summary: sends a performance summary via Telegram at 00:00 UTC each day.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class SignalMonitor:
    """Monitor signal generation and send periodic summaries.

    Runs a lightweight daemon thread that checks every 60 seconds for:
    - No signal generated within ``no_signal_threshold_hours`` during market hours
    - Daily summary trigger at 00:00 UTC

    Args:
        alerting_manager: ``live_trading.alerting.AlertManager`` instance for sending alerts.
        signal_tracker: Optional ``SignalTracker`` for gathering performance stats.
        kill_switch: Optional ``KillSwitch`` for reading current halt level.
        no_signal_threshold_hours: Hours without a signal before alerting (default 2).
    """

    def __init__(
        self,
        alerting_manager,
        signal_tracker=None,
        kill_switch=None,
        no_signal_threshold_hours: int = 2,
    ):
        self._alerting = alerting_manager
        self._tracker = signal_tracker
        self._kill_switch = kill_switch
        self._no_signal_threshold = timedelta(hours=no_signal_threshold_hours)

        # Timestamps
        self._last_signal_time: datetime = datetime.now(timezone.utc)
        self._last_daily_summary_date: datetime | None = None
        self._last_no_signal_alert: datetime | None = None

        # Thread control
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._check_interval = 60  # seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="signal-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("SignalMonitor started")

    def stop(self) -> None:
        """Gracefully stop the monitoring thread."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("SignalMonitor stopped")

    def record_signal(self) -> None:
        """Called when a new signal is generated — resets the no-signal timer."""
        self._last_signal_time = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Main monitoring loop — runs every ``_check_interval`` seconds."""
        while self._running:
            try:
                self._check_no_signal()
                self._check_daily_summary()
            except Exception:
                logger.exception("SignalMonitor check failed")
            self._stop_event.wait(timeout=self._check_interval)

    # ------------------------------------------------------------------
    # No-signal detection
    # ------------------------------------------------------------------

    def _check_no_signal(self) -> None:
        """Alert if no signal has been generated within the threshold during market hours."""
        now = datetime.now(timezone.utc)

        if not self._is_market_hours(now):
            return

        elapsed = now - self._last_signal_time
        if elapsed < self._no_signal_threshold:
            return

        # Deduplicate: don't re-alert within the same threshold window
        if (
            self._last_no_signal_alert is not None
            and (now - self._last_no_signal_alert) < self._no_signal_threshold
        ):
            return

        hours = elapsed.total_seconds() / 3600
        self._alerting.warning(
            "No Signal Generated",
            f"No trading signal generated in {hours:.1f} hours during market hours. "
            "Bot may be stuck or model not producing actionable signals.",
            hours_since_last=f"{hours:.1f}",
            last_signal=self._last_signal_time.isoformat(),
        )
        self._last_no_signal_alert = now
        logger.warning("No-signal alert fired: %.1f hours since last signal", hours)

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    def _check_daily_summary(self) -> None:
        """Send daily summary at 00:00 UTC if not already sent today."""
        now = datetime.now(timezone.utc)
        today = now.date()

        # Only fire once per calendar day, and only after midnight
        if self._last_daily_summary_date == today:
            return
        if now.hour != 0:
            return

        self._send_daily_summary()
        self._last_daily_summary_date = today

    def _send_daily_summary(self) -> None:
        """Gather stats and send daily performance summary."""
        trades = 0
        profit = 0.0
        win_rate = 0.0
        max_dd = 0.0
        var_95 = None
        correlation_regime = None
        kill_switch_status = None

        # Pull stats from tracker (previous day)
        if self._tracker is not None:
            try:
                summary = self._tracker.get_performance_summary(days=1)
                trades = summary.get("total", 0)
                profit = summary.get("cumulative_pnl", 0.0)
                win_rate = summary.get("win_rate", 0.0)
                max_dd = summary.get("max_drawdown_pct", 0.0)
            except Exception:
                logger.exception("Failed to get tracker summary for daily report")

        # Pull kill switch status
        if self._kill_switch is not None:
            try:
                kill_switch_status = self._kill_switch.halt_level.name
            except Exception:
                logger.exception("Failed to read kill switch status")

        self._alerting.daily_summary(
            trades=trades,
            profit=profit,
            win_rate=win_rate,
            max_dd=max_dd,
            var_95=var_95,
            correlation_regime=correlation_regime,
            kill_switch_status=kill_switch_status,
        )
        logger.info("Daily summary sent")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_market_hours(dt: datetime) -> bool:
        """Return True if *dt* falls within forex market hours.

        Forex markets are open from Sunday 22:00 UTC to Friday 22:00 UTC.
        """
        weekday = dt.weekday()  # Mon=0, Sun=6
        hour = dt.hour

        # Saturday: always closed
        if weekday == 5:
            return False

        # Sunday: open only from 22:00 onward
        if weekday == 6:
            return hour >= 22

        # Friday: open only until 22:00
        if weekday == 4:
            return hour < 22

        # Mon-Thu: always open
        return True
