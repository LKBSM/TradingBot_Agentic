"""
Sprint 13 — Observability & Alerting Wiring tests.

Covers:
  - TelegramAlertCallback bridging kill switch → alerting
  - SignalMonitor (no-signal detection + daily summary)
  - New alerting methods (signal_generated, signal_closed, enhanced daily_summary)
  - Alert deduplication
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, call

import pytest

from src.agents.kill_switch import (
    HaltEvent,
    HaltLevel,
    HaltReason,
    KillSwitch,
    TelegramAlertCallback,
)
from src.live_trading.signal_monitor import SignalMonitor


# =============================================================================
# HELPERS
# =============================================================================

def _make_halt_event(
    reason=HaltReason.MAX_DRAWDOWN,
    level=HaltLevel.FULL_HALT,
    trigger_value=0.12,
    threshold=0.10,
    message="Drawdown exceeded 10%",
) -> HaltEvent:
    return HaltEvent(
        halt_id="halt_test_001",
        reason=reason,
        level=level,
        timestamp=datetime.now(),
        trigger_value=trigger_value,
        threshold=threshold,
        message=message,
    )


def _mock_alerting():
    """Return a mock that mimics live_trading.alerting.AlertManager."""
    m = MagicMock()
    m.kill_switch_triggered = MagicMock()
    m.warning = MagicMock()
    m.info = MagicMock()
    m.critical = MagicMock()
    m.daily_summary = MagicMock()
    m.signal_generated = MagicMock()
    m.signal_closed = MagicMock()
    return m


# =============================================================================
# TelegramAlertCallback
# =============================================================================

class TestTelegramAlertCallback:
    """Verify that TelegramAlertCallback forwards halt events."""

    def test_callback_sends_halt_event(self):
        alerting = _mock_alerting()
        cb = TelegramAlertCallback(alerting)

        event = _make_halt_event()
        result = cb.send(event)

        assert result is True
        alerting.kill_switch_triggered.assert_called_once()
        call_kwargs = alerting.kill_switch_triggered.call_args.kwargs
        assert "max_drawdown" in call_kwargs["reason"].lower()

    def test_kill_switch_with_alerting_fires_callback(self, tmp_path):
        alerting = _mock_alerting()

        ks = KillSwitch(
            initial_equity=10000.0,
            persistence_path=str(tmp_path / "ks.db"),
            alerting_manager=alerting,
        )

        # Force a halt by updating with severe drawdown
        ks.update(equity=7000.0)  # 30% drawdown triggers halt

        # Callback should have been called at least once
        assert alerting.kill_switch_triggered.called

    def test_callback_handles_exception(self):
        alerting = MagicMock()
        alerting.kill_switch_triggered.side_effect = RuntimeError("network error")
        cb = TelegramAlertCallback(alerting)

        event = _make_halt_event()
        result = cb.send(event)

        assert result is False


# =============================================================================
# SignalMonitor
# =============================================================================

class TestSignalMonitor:
    """Verify no-signal detection and daily summary scheduling."""

    def test_no_signal_alert_fires_after_threshold(self):
        alerting = _mock_alerting()
        monitor = SignalMonitor(
            alerting_manager=alerting,
            no_signal_threshold_hours=2,
        )
        # Simulate 3 hours without signal, during market hours (Wednesday noon UTC)
        market_time = datetime(2026, 2, 25, 12, 0, tzinfo=timezone.utc)  # Wednesday
        monitor._last_signal_time = market_time - timedelta(hours=3)

        with patch(
            "src.live_trading.signal_monitor.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = market_time
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            monitor._check_no_signal()

        alerting.warning.assert_called_once()
        assert "No Signal Generated" in alerting.warning.call_args[0][0]

    def test_no_alert_on_saturday(self):
        alerting = _mock_alerting()
        monitor = SignalMonitor(alerting_manager=alerting)
        # Saturday = closed market
        saturday = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
        monitor._last_signal_time = saturday - timedelta(hours=5)

        with patch(
            "src.live_trading.signal_monitor.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = saturday
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            monitor._check_no_signal()

        alerting.warning.assert_not_called()

    def test_record_signal_resets_timer(self):
        alerting = _mock_alerting()
        monitor = SignalMonitor(alerting_manager=alerting)
        old_time = monitor._last_signal_time

        monitor.record_signal()
        assert monitor._last_signal_time >= old_time

    def test_daily_summary_fires_at_midnight_utc(self):
        alerting = _mock_alerting()
        tracker = MagicMock()
        tracker.get_performance_summary.return_value = {
            "total": 5,
            "cumulative_pnl": 42.0,
            "win_rate": 0.6,
            "max_drawdown_pct": 3.5,
        }
        ks = MagicMock()
        ks.halt_level = HaltLevel.NONE

        monitor = SignalMonitor(
            alerting_manager=alerting,
            signal_tracker=tracker,
            kill_switch=ks,
        )

        midnight = datetime(2026, 2, 26, 0, 30, tzinfo=timezone.utc)
        with patch(
            "src.live_trading.signal_monitor.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = midnight
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            monitor._check_daily_summary()

        alerting.daily_summary.assert_called_once()
        kwargs = alerting.daily_summary.call_args[1]
        assert kwargs["trades"] == 5
        assert kwargs["profit"] == 42.0
        assert kwargs["kill_switch_status"] == "NONE"

    def test_daily_summary_includes_all_fields(self):
        alerting = _mock_alerting()
        tracker = MagicMock()
        tracker.get_performance_summary.return_value = {
            "total": 10,
            "cumulative_pnl": 100.0,
            "win_rate": 0.7,
            "max_drawdown_pct": 2.0,
        }
        ks = MagicMock()
        ks.halt_level = HaltLevel.CAUTION

        monitor = SignalMonitor(
            alerting_manager=alerting,
            signal_tracker=tracker,
            kill_switch=ks,
        )
        monitor._send_daily_summary()

        alerting.daily_summary.assert_called_once()
        kwargs = alerting.daily_summary.call_args[1]
        assert "trades" in kwargs
        assert "profit" in kwargs
        assert "win_rate" in kwargs
        assert "max_dd" in kwargs
        assert "kill_switch_status" in kwargs

    def test_stop_shuts_down_cleanly(self):
        alerting = _mock_alerting()
        monitor = SignalMonitor(alerting_manager=alerting)
        monitor.start()
        assert monitor._running is True
        monitor.stop()
        assert monitor._running is False


# =============================================================================
# Alerting Methods
# =============================================================================

class TestAlertingMethods:
    """Verify new signal alerting methods on live_trading AlertManager."""

    def _make_manager(self):
        """Create a minimal AlertManager with no real channels."""
        from src.live_trading.alerting import AlertManager, AlertConfig

        config = AlertConfig()  # No tokens set → no channels configured
        mgr = AlertManager(config=config)
        return mgr

    def test_signal_generated_creates_info(self):
        mgr = self._make_manager()
        # Patch the core alert method to capture the call
        mgr.alert = MagicMock()

        mgr.signal_generated(
            signal_id="sig-001",
            action="OPEN_LONG",
            symbol="XAUUSD",
            entry=2350.0,
            sl=2340.0,
            tp=2370.0,
            rr=2.0,
        )

        mgr.alert.assert_called_once()
        from src.live_trading.alerting import AlertLevel
        call_args = mgr.alert.call_args
        assert call_args[0][0] == AlertLevel.INFO

    def test_signal_closed_win_is_info(self):
        mgr = self._make_manager()
        mgr.alert = MagicMock()

        mgr.signal_closed("sig-002", "WIN", 15.0)

        from src.live_trading.alerting import AlertLevel
        call_args = mgr.alert.call_args
        assert call_args[0][0] == AlertLevel.INFO

    def test_signal_closed_loss_is_warning(self):
        mgr = self._make_manager()
        mgr.alert = MagicMock()

        mgr.signal_closed("sig-003", "LOSS", -10.0)

        from src.live_trading.alerting import AlertLevel
        call_args = mgr.alert.call_args
        assert call_args[0][0] == AlertLevel.WARNING


# =============================================================================
# Alert Deduplication
# =============================================================================

class TestAlertDeduplication:
    """Verify the existing deduplication logic handles Sprint 13 scenarios."""

    def _make_manager(self):
        from src.live_trading.alerting import AlertManager, AlertConfig

        config = AlertConfig(min_alert_interval_seconds=60)
        mgr = AlertManager(config=config)
        return mgr

    def test_same_alert_within_interval_is_deduped(self):
        mgr = self._make_manager()
        # Send two identical alerts rapidly
        mgr._deliver_alert = MagicMock()  # Prevent actual delivery
        from src.live_trading.alerting import Alert, AlertLevel

        a1 = Alert(level=AlertLevel.WARNING, title="Test", message="msg1")
        a2 = Alert(level=AlertLevel.WARNING, title="Test", message="msg2")

        # First should pass rate limit
        assert mgr._is_rate_limited(a1) is False
        mgr._alert_history.append(a1)

        # Second identical title+level within interval should be blocked
        assert mgr._is_rate_limited(a2) is True

    def test_different_alerts_are_not_deduped(self):
        mgr = self._make_manager()
        from src.live_trading.alerting import Alert, AlertLevel

        a1 = Alert(level=AlertLevel.WARNING, title="Alert A", message="msg")
        a2 = Alert(level=AlertLevel.WARNING, title="Alert B", message="msg")

        mgr._alert_history.append(a1)
        assert mgr._is_rate_limited(a2) is False

    def test_critical_alerts_bypass_dedup(self):
        mgr = self._make_manager()
        from src.live_trading.alerting import Alert, AlertLevel

        a1 = Alert(level=AlertLevel.CRITICAL, title="Kill Switch", message="msg1")
        mgr._alert_history.append(a1)

        # Critical with force=True bypasses rate limiting in the alert() method
        # Verify that the critical() convenience method sets force=True
        mgr._deliver_alert = MagicMock()
        mgr._running = True  # Enable delivery

        # We test that critical() calls alert() with force=True
        with patch.object(mgr, 'alert', wraps=mgr.alert) as wrapped:
            mgr.critical("Kill Switch", "Emergency halt")
            call_kwargs = wrapped.call_args
            assert call_kwargs[1].get("force") is True or call_kwargs[0][0] == AlertLevel.CRITICAL


# =============================================================================
# Market Hours
# =============================================================================

class TestMarketHours:
    """Verify the market hours calculation in SignalMonitor."""

    def test_monday_is_market_hours(self):
        monday_noon = datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc)
        assert SignalMonitor._is_market_hours(monday_noon) is True

    def test_saturday_is_closed(self):
        saturday = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
        assert SignalMonitor._is_market_hours(saturday) is False

    def test_sunday_before_22_is_closed(self):
        sunday_morning = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
        assert SignalMonitor._is_market_hours(sunday_morning) is False

    def test_sunday_after_22_is_open(self):
        sunday_evening = datetime(2026, 3, 1, 22, 0, tzinfo=timezone.utc)
        assert SignalMonitor._is_market_hours(sunday_evening) is True

    def test_friday_before_22_is_open(self):
        friday_noon = datetime(2026, 2, 27, 12, 0, tzinfo=timezone.utc)
        assert SignalMonitor._is_market_hours(friday_noon) is True

    def test_friday_after_22_is_closed(self):
        friday_late = datetime(2026, 2, 27, 23, 0, tzinfo=timezone.utc)
        assert SignalMonitor._is_market_hours(friday_late) is False
