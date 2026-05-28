"""DG-052 — Cost-alert watcher tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.intelligence.rag.cost_alert_watcher import (
    CostAlertWatcher,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_THRESHOLD_USD,
    DiscordWebhookNotifier,
    LoggingNotifier,
)


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------

def test_threshold_must_be_positive():
    with pytest.raises(ValueError):
        CostAlertWatcher(cost_provider=lambda: 0.0, threshold_usd=0)


def test_cooldown_must_be_non_negative():
    with pytest.raises(ValueError):
        CostAlertWatcher(cost_provider=lambda: 0.0, cooldown_seconds=-1)


def test_default_threshold_constants_export():
    assert DEFAULT_THRESHOLD_USD > 0
    assert DEFAULT_COOLDOWN_SECONDS > 0


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

def test_below_threshold_does_not_fire():
    notify = MagicMock()
    w = CostAlertWatcher(cost_provider=lambda: 1.0, threshold_usd=5.0, notify=notify)
    assert w.check() is False
    notify.assert_not_called()
    assert w.state == "below"
    assert w.alerts_fired == 0


def test_at_or_above_threshold_fires_once():
    notify = MagicMock()
    w = CostAlertWatcher(cost_provider=lambda: 5.0, threshold_usd=5.0, notify=notify)
    assert w.check() is True
    notify.assert_called_once()
    assert w.state == "alerting"
    assert w.alerts_fired == 1


def test_cooldown_suppresses_re_fire_within_window():
    notify = MagicMock()
    fake_time = [1000.0]
    w = CostAlertWatcher(
        cost_provider=lambda: 10.0,
        threshold_usd=5.0,
        notify=notify,
        cooldown_seconds=300.0,
        clock=lambda: fake_time[0],
    )
    # First crossing fires
    assert w.check() is True
    # Within cooldown — does NOT fire
    fake_time[0] += 100  # 100s < cooldown 300s
    assert w.check() is False
    assert notify.call_count == 1


def test_cooldown_expires_then_re_fires():
    notify = MagicMock()
    fake_time = [1000.0]
    w = CostAlertWatcher(
        cost_provider=lambda: 10.0,
        threshold_usd=5.0,
        notify=notify,
        cooldown_seconds=300.0,
        clock=lambda: fake_time[0],
    )
    w.check()  # 1st fire
    fake_time[0] += 301
    assert w.check() is True
    assert notify.call_count == 2


def test_drop_below_then_back_above_triggers_again():
    notify = MagicMock()
    fake_time = [1000.0]
    spend = [10.0]
    w = CostAlertWatcher(
        cost_provider=lambda: spend[0],
        threshold_usd=5.0,
        notify=notify,
        cooldown_seconds=10.0,
        clock=lambda: fake_time[0],
    )
    # First crossing fires
    w.check()
    # Drop back below
    spend[0] = 1.0
    w.check()
    assert w.state == "below"
    # Back above (after cooldown expires)
    fake_time[0] += 20.0
    spend[0] = 10.0
    assert w.check() is True
    assert notify.call_count == 2


def test_notifier_exception_does_not_break_watcher():
    """If the notifier raises, the watcher must keep working."""
    notify = MagicMock(side_effect=RuntimeError("Discord down"))
    w = CostAlertWatcher(
        cost_provider=lambda: 10.0,
        threshold_usd=5.0,
        notify=notify,
    )
    # Returns True (we did try to fire) but does not raise
    assert w.check() is True
    assert w.alerts_fired == 1


def test_reset_returns_to_below_state():
    notify = MagicMock()
    w = CostAlertWatcher(cost_provider=lambda: 10.0, threshold_usd=5.0, notify=notify)
    w.check()
    assert w.state == "alerting"
    w.reset()
    assert w.state == "below"


def test_message_includes_threshold_and_observed_value():
    captured = []
    w = CostAlertWatcher(
        cost_provider=lambda: 7.5,
        threshold_usd=5.0,
        notify=lambda msg: captured.append(msg),
        label="LLM",
    )
    w.check()
    assert captured
    assert "5.0000" in captured[0]
    assert "7.5000" in captured[0]
    assert "LLM" in captured[0]


# ---------------------------------------------------------------------------
# Notifier adapters
# ---------------------------------------------------------------------------

def test_logging_notifier_emits_to_module_logger(caplog):
    notifier = LoggingNotifier()
    with caplog.at_level("WARNING"):
        notifier("hello")
    assert any("hello" in r.message for r in caplog.records)


def test_discord_webhook_requires_url():
    with pytest.raises(ValueError):
        DiscordWebhookNotifier("")


def test_discord_webhook_swallows_network_error(monkeypatch):
    """A Discord outage must NOT bubble up to the watcher."""
    import urllib.error
    notifier = DiscordWebhookNotifier("https://discord.example.com/webhook")

    def fake_urlopen(*_args, **_kwargs):
        raise urllib.error.URLError("name resolution failure")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    # Should not raise
    notifier("hello")
