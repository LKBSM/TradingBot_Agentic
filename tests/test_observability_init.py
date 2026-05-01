"""Tests for src/performance/observability.py (INFRA-1.2 sprint).

Per DoD: Sentry captures test error, /metrics returns non-empty payload,
JSON logs parsable. Tests focus on the contract (DSN handling, metric
registration, scrubber behaviour) — actual Sentry network calls are not
made (would require a valid DSN).

Note: filename is `test_observability_init.py` to avoid clashing with the
pre-existing `test_observability.py` (Sprint 13 alerting wiring tests).
"""

from __future__ import annotations

import pytest

from src.performance.metrics import MetricsRegistry
from src.performance.observability import (
    STANDARD_METRIC_NAMES,
    StandardMetrics,
    _scrub_pii,
    init_observability,
    init_sentry,
    register_standard_metrics,
)


# ---------------------------------------------------------------------------
# Standard metrics registration
# ---------------------------------------------------------------------------


def test_standard_metric_names_includes_three_plan_mandated():
    assert "signals_generated_total" in STANDARD_METRIC_NAMES
    assert "llm_latency_seconds" in STANDARD_METRIC_NAMES
    assert "circuit_breaker_open_total" in STANDARD_METRIC_NAMES


def test_register_standard_metrics_returns_handles():
    registry = MetricsRegistry(prefix="t1")
    sm = register_standard_metrics(registry)
    assert isinstance(sm, StandardMetrics)
    assert hasattr(sm.signals_generated_total, "inc")
    assert hasattr(sm.circuit_breaker_open_total, "inc")
    assert hasattr(sm.llm_latency_seconds, "observe")


def test_register_standard_metrics_idempotent():
    """Re-registering must not raise — refetches the existing series."""
    registry = MetricsRegistry(prefix="t2")
    sm1 = register_standard_metrics(registry)
    sm2 = register_standard_metrics(registry)
    assert sm1.signals_generated_total is sm2.signals_generated_total


def test_metrics_appear_in_prometheus_output():
    registry = MetricsRegistry(prefix="testapp")
    sm = register_standard_metrics(registry)
    sm.signals_generated_total.inc(5)
    sm.circuit_breaker_open_total.inc(1)

    output = registry.to_prometheus()
    assert "testapp_signals_generated_total" in output
    assert "testapp_circuit_breaker_open_total" in output
    assert "testapp_llm_latency_seconds" in output


# ---------------------------------------------------------------------------
# Sentry init (no actual network)
# ---------------------------------------------------------------------------


def test_init_sentry_returns_false_when_dsn_unset(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert init_sentry() is False


def test_init_sentry_returns_false_when_dsn_empty(monkeypatch):
    monkeypatch.setenv("SENTRY_DSN", "   ")
    assert init_sentry() is False


def test_init_sentry_does_not_raise_on_invalid_dsn(monkeypatch):
    """Sentry init with a malformed DSN should warn but not crash."""
    monkeypatch.setenv("SENTRY_DSN", "not-a-valid-dsn")
    try:
        init_sentry()
    except Exception as exc:
        pytest.fail(f"init_sentry should not raise on invalid DSN, got {exc}")


# ---------------------------------------------------------------------------
# PII scrubber
# ---------------------------------------------------------------------------


def test_scrub_pii_drops_event_with_anthropic_key_marker():
    event = {
        "message": "API call failed",
        "extra": {"raw": "ANTHROPIC_API_KEY=sk-ant-secret123"},
    }
    assert _scrub_pii(event, hint={}) is None


def test_scrub_pii_drops_event_with_fred_key_marker():
    event = {"extra": "log line: FRED_API_KEY=abc123"}
    assert _scrub_pii(event, hint={}) is None


def test_scrub_pii_drops_event_with_telegram_token_marker():
    event = {"breadcrumbs": [{"message": "TELEGRAM_BOT_TOKEN=12345:abc"}]}
    assert _scrub_pii(event, hint={}) is None


def test_scrub_pii_passes_clean_event():
    event = {"message": "Signal generated for XAUUSD at 2350"}
    assert _scrub_pii(event, hint={}) is not None
    assert _scrub_pii(event, hint={}) == event


# ---------------------------------------------------------------------------
# Top-level boot helper
# ---------------------------------------------------------------------------


def test_init_observability_returns_registry_with_metrics(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    registry = init_observability(service="boot_test")
    assert isinstance(registry, MetricsRegistry)
    out = registry.to_prometheus()
    assert "boot_test_signals_generated_total" in out
    assert "boot_test_circuit_breaker_open_total" in out
    assert "boot_test_llm_latency_seconds" in out


def test_init_observability_metrics_prefix_overrides_service():
    registry = init_observability(service="ignored", metrics_prefix="custom_pref")
    out = registry.to_prometheus()
    assert "custom_pref_signals_generated_total" in out


def test_init_observability_payload_is_non_empty():
    """eval_16 finding #1: /metrics payload was empty in prod. After init
    the registry must produce a non-empty Prometheus exposition."""
    registry = init_observability(service="empty_check")
    out = registry.to_prometheus()
    assert len(out) > 100
