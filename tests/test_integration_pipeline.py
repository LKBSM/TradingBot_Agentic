"""
Sprint 12 — End-to-end integration tests for the signal pipeline.

Tests the critical path:
  Signal generation → Risk check → Kill switch → API publish →
  Outcome tracking → Dashboard metrics → Prometheus update
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import KeyStore
from src.api.signal_store import SignalRecord, SignalStore
from src.api.signal_tracker import SignalTracker
from src.performance.metrics import MetricsRegistry, create_trading_metrics


# Disable TESTING_MODE for pipeline auth tests
@pytest.fixture(autouse=True)
def _disable_testing_mode():
    with patch("src.api.auth.TESTING_MODE", False), \
         patch("src.api.routes.health.TESTING_MODE", False):
        yield


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture()
def pipeline_db(tmp_path):
    return str(tmp_path / "pipeline.db")


@pytest.fixture()
def pipeline(pipeline_db, tmp_path):
    """Full pipeline: store + tracker + metrics + API client."""
    store = SignalStore(db_path=pipeline_db)
    tracker = SignalTracker(db_path=pipeline_db)
    registry = MetricsRegistry(prefix="test_pipe")
    metrics = create_trading_metrics(registry)
    ks_db = str(tmp_path / "keys.db")
    key_store = KeyStore(db_path=ks_db)

    app = create_app(
        signal_store=store,
        signal_tracker=tracker,
        metrics_registry=registry,
        key_store=key_store,
    )
    client = TestClient(app)
    api_key = key_store.create_key("integration-test")["api_key"]
    headers = {"X-API-Key": api_key}

    return {
        "store": store,
        "tracker": tracker,
        "registry": registry,
        "metrics": metrics,
        "client": client,
        "headers": headers,
    }


# =============================================================================
# FULL PIPELINE TESTS
# =============================================================================

class TestSignalPipeline:
    """End-to-end: publish signal → read via API → close → verify dashboard."""

    def test_publish_and_read_current(self, pipeline):
        """Signal published by trading loop is immediately readable via API."""
        store = pipeline["store"]
        client = pipeline["client"]
        headers = pipeline["headers"]

        signal = SignalRecord(
            signal_id="pipe-001",
            action="OPEN_LONG",
            symbol="XAUUSD",
            entry_price=2350.0,
            stop_loss=2340.0,
            take_profit=2370.0,
            rr_ratio=2.0,
            created_at=datetime.now().isoformat(),
        )
        store.publish(signal)

        resp = client.get("/api/v1/signals/current", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["signal_id"] == "pipe-001"
        assert body["action"] == "OPEN_LONG"

    def test_outcome_reflected_in_history(self, pipeline):
        """After outcome update, history endpoint shows the result."""
        store = pipeline["store"]
        client = pipeline["client"]
        headers = pipeline["headers"]

        signal = SignalRecord(
            signal_id="pipe-002",
            action="OPEN_SHORT",
            symbol="XAUUSD",
            entry_price=2370.0,
            stop_loss=2380.0,
            take_profit=2350.0,
            rr_ratio=2.0,
            created_at=datetime.now().isoformat(),
        )
        store.publish(signal)
        store.update_outcome("pipe-002", "WIN", 20.0)

        resp = client.get("/api/v1/signals/history", headers=headers)
        assert resp.status_code == 200
        sig = resp.json()["signals"][0]
        assert sig["outcome"] == "WIN"
        assert sig["pnl_pips"] == pytest.approx(20.0)

    def test_dashboard_summary_reflects_outcomes(self, pipeline):
        """Dashboard summary aggregates closed signals correctly."""
        store = pipeline["store"]
        client = pipeline["client"]
        headers = pipeline["headers"]

        now = datetime.now()
        for i, (outcome, pnl) in enumerate([
            ("WIN", 15.0), ("WIN", 25.0), ("LOSS", -10.0),
        ]):
            sid = f"pipe-sum-{i}"
            store.publish(SignalRecord(
                signal_id=sid, action="OPEN_LONG", symbol="XAUUSD",
                entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0,
                rr_ratio=2.0,
                created_at=(now - timedelta(hours=i + 1)).isoformat(),
            ))
            store.update_outcome(sid, outcome, pnl, now.isoformat())

        resp = client.get("/api/v1/dashboard/summary", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_signals"] == 3
        assert body["winning"] == 2
        assert body["losing"] == 1
        assert body["cumulative_pnl"] == pytest.approx(30.0)

    def test_equity_curve_matches_outcomes(self, pipeline):
        """Equity curve shows cumulative PnL consistent with outcomes."""
        store = pipeline["store"]
        client = pipeline["client"]
        headers = pipeline["headers"]

        now = datetime.now()
        pnls = [10.0, -5.0, 20.0]
        for i, pnl in enumerate(pnls):
            sid = f"pipe-eq-{i}"
            outcome = "WIN" if pnl > 0 else "LOSS"
            closed = (now - timedelta(days=3 - i)).isoformat()
            store.publish(SignalRecord(
                signal_id=sid, action="OPEN_LONG", symbol="XAUUSD",
                entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0,
                rr_ratio=2.0,
                created_at=(now - timedelta(days=3 - i, hours=2)).isoformat(),
            ))
            store.update_outcome(sid, outcome, pnl, closed)

        resp = client.get("/api/v1/dashboard/equity-curve", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["points"]) == 3
        assert body["points"][-1]["cumulative_pnl"] == pytest.approx(25.0)
        assert body["current_cumulative_pnl"] == pytest.approx(25.0)

    def test_metrics_counter_updates(self, pipeline):
        """Metrics counters track signal events."""
        metrics = pipeline["metrics"]

        metrics["signals_published"].inc()
        metrics["signals_published"].inc()
        metrics["signals_closed_win"].inc()
        metrics["signals_closed_loss"].inc()

        assert metrics["signals_published"].get() == 2
        assert metrics["signals_closed_win"].get() == 1
        assert metrics["signals_closed_loss"].get() == 1

    def test_metrics_gauges_update(self, pipeline):
        """Performance gauges can be set and read."""
        metrics = pipeline["metrics"]

        metrics["current_win_rate_30d"].set(0.65)
        metrics["current_sharpe_30d"].set(1.8)
        metrics["current_profit_factor_30d"].set(2.1)
        metrics["current_cumulative_pnl"].set(150.0)

        assert metrics["current_win_rate_30d"].get() == pytest.approx(0.65)
        assert metrics["current_sharpe_30d"].get() == pytest.approx(1.8)
        assert metrics["current_profit_factor_30d"].get() == pytest.approx(2.1)
        assert metrics["current_cumulative_pnl"].get() == pytest.approx(150.0)

    def test_pnl_histogram_records(self, pipeline):
        """Signal PnL histogram accepts and summarizes values."""
        metrics = pipeline["metrics"]

        for pnl in [15.0, -8.0, 30.0, -20.0, 5.0]:
            metrics["signal_pnl_pips"].observe(pnl)

        summary = metrics["signal_pnl_pips"].get_summary()
        assert summary["count"] == 5
        assert summary["sum"] == pytest.approx(22.0)

    def test_prometheus_exposes_signal_metrics(self, pipeline):
        """Prometheus endpoint includes signal-specific metrics."""
        client = pipeline["client"]
        metrics = pipeline["metrics"]

        metrics["signals_published"].inc()
        metrics["current_win_rate_30d"].set(0.55)

        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "signals_published_total" in text
        assert "current_win_rate_30d" in text


# =============================================================================
# PIPELINE ERROR HANDLING TESTS
# =============================================================================

class TestPipelineResilience:
    """Verify the pipeline handles edge cases gracefully."""

    def test_dashboard_with_no_closed_signals(self, pipeline):
        """Dashboard returns zeros when no signals are closed."""
        store = pipeline["store"]
        client = pipeline["client"]
        headers = pipeline["headers"]

        store.publish(SignalRecord(
            signal_id="open-only", action="OPEN_LONG", symbol="XAUUSD",
            entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0,
            rr_ratio=2.0, created_at=datetime.now().isoformat(),
        ))

        resp = client.get("/api/v1/dashboard/summary", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["total_signals"] == 0

    def test_auth_required_on_all_subscriber_endpoints(self, pipeline):
        """All subscriber endpoints reject requests without API key."""
        client = pipeline["client"]

        endpoints = [
            "/api/v1/signals/current",
            "/api/v1/signals/history",
            "/api/v1/dashboard/summary",
            "/api/v1/dashboard/equity-curve",
            "/api/v1/operator/metrics",
            "/api/v1/operator/risk",
        ]
        for endpoint in endpoints:
            resp = client.get(endpoint)
            assert resp.status_code == 401, f"{endpoint} should require auth"

    def test_health_and_prometheus_are_public(self, pipeline):
        """Health and Prometheus remain public (no auth required)."""
        client = pipeline["client"]

        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200

    def test_multiple_signals_sequential_consistency(self, pipeline):
        """Publishing multiple signals maintains consistency."""
        store = pipeline["store"]
        client = pipeline["client"]
        headers = pipeline["headers"]

        for i in range(10):
            store.publish(SignalRecord(
                signal_id=f"seq-{i:03d}",
                action="OPEN_LONG" if i % 2 == 0 else "OPEN_SHORT",
                symbol="XAUUSD",
                entry_price=2350.0 + i,
                stop_loss=2340.0,
                take_profit=2370.0,
                rr_ratio=2.0,
                created_at=datetime.now().isoformat(),
            ))

        current = client.get("/api/v1/signals/current", headers=headers).json()
        assert current["signal_id"] == "seq-009"

        history = client.get("/api/v1/signals/history", headers=headers).json()
        assert history["total"] == 10

    def test_tracker_and_store_share_db_correctly(self, pipeline):
        """Tracker reads what store writes (shared DB)."""
        store = pipeline["store"]
        tracker = pipeline["tracker"]

        now = datetime.now()
        store.publish(SignalRecord(
            signal_id="share-1", action="OPEN_LONG", symbol="XAUUSD",
            entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0,
            rr_ratio=2.0,
            created_at=(now - timedelta(hours=2)).isoformat(),
        ))
        store.update_outcome("share-1", "WIN", 10.0, now.isoformat())

        summary = tracker.get_performance_summary(days=30)
        assert summary["total"] == 1
        assert summary["won"] == 1

        curve = tracker.get_equity_curve(days=30)
        assert len(curve) == 1
        assert curve[0]["cumulative_pnl"] == pytest.approx(10.0)
