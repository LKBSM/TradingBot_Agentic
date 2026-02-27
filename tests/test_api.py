"""
Sprint 9 — FastAPI Signal Delivery API tests.

Covers:
  - Signal endpoints (current, history, no-leak, pagination, outcomes)
  - Health endpoints (v1 + Docker compat)
  - Operator endpoints (metrics, risk, kill-switch, safe defaults)
  - Prometheus scrape endpoint
  - SignalStore unit tests (publish/read, history, outcome, thread safety)
  - Middleware tests (CORS, 404)
"""

from __future__ import annotations

import os
import tempfile
import threading
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import KeyStore
from src.api.signal_store import SignalRecord, SignalStore
from src.performance.metrics import MetricsRegistry


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture()
def tmp_db(tmp_path):
    """Return a temporary SQLite path."""
    return str(tmp_path / "test_signals.db")


@pytest.fixture()
def store(tmp_db):
    return SignalStore(db_path=tmp_db)


@pytest.fixture()
def sample_signal() -> SignalRecord:
    return SignalRecord(
        signal_id="sig-001",
        action="OPEN_LONG",
        symbol="XAUUSD",
        entry_price=2350.50,
        stop_loss=2340.00,
        take_profit=2370.00,
        rr_ratio=1.86,
        created_at=datetime.now().isoformat(),
    )


@pytest.fixture()
def registry():
    return MetricsRegistry(prefix="test")


@pytest.fixture()
def tmp_keys_db(tmp_path):
    """Return a temporary SQLite path for API keys."""
    return str(tmp_path / "test_api_keys.db")


@pytest.fixture()
def key_store(tmp_keys_db):
    return KeyStore(db_path=tmp_keys_db)


@pytest.fixture()
def test_api_key(key_store):
    """Pre-create a test key and return the raw key."""
    return key_store.create_key("test-subscriber")["api_key"]


@pytest.fixture()
def auth_headers(test_api_key):
    """Standard auth headers for signal/operator requests."""
    return {"X-API-Key": test_api_key}


@pytest.fixture()
def client(tmp_db, registry, key_store):
    """TestClient with a fresh SignalStore + MetricsRegistry + KeyStore."""
    signal_store = SignalStore(db_path=tmp_db)
    app = create_app(
        signal_store=signal_store,
        metrics_registry=registry,
        key_store=key_store,
    )
    return TestClient(app)


@pytest.fixture()
def client_with_kill_switch(tmp_db, registry, key_store):
    """TestClient with a mock KillSwitch wired in."""
    signal_store = SignalStore(db_path=tmp_db)
    ks = MagicMock()
    ks.halt_level.value = 0
    ks.is_trading_allowed.return_value = True
    ks.get_status.return_value = {
        "halt_level": "NONE",
        "halt_level_value": 0,
        "is_halted": False,
        "is_trading_allowed": True,
        "tracking": {
            "equity": 100_000,
            "peak_equity": 105_000,
            "daily_pnl": -200,
            "drawdown_pct": 0.047,
        },
        "breakers": {},
        "recovery": {},
        "config": {},
    }
    app = create_app(
        signal_store=signal_store,
        metrics_registry=registry,
        kill_switch=ks,
        key_store=key_store,
    )
    return TestClient(app)


# =============================================================================
# SIGNAL STORE UNIT TESTS
# =============================================================================

class TestSignalStore:
    def test_publish_and_get_current(self, store, sample_signal):
        assert store.get_current() is None
        store.publish(sample_signal)
        cur = store.get_current()
        assert cur is not None
        assert cur.signal_id == "sig-001"
        assert cur.action == "OPEN_LONG"

    def test_history_persistence(self, store):
        for i in range(5):
            store.publish(
                SignalRecord(
                    signal_id=f"sig-{i:03d}",
                    action="OPEN_LONG",
                    symbol="XAUUSD",
                    entry_price=2350.0 + i,
                    stop_loss=2340.0,
                    take_profit=2370.0,
                    rr_ratio=1.5,
                    created_at=f"2025-01-01T00:00:{i:02d}",
                )
            )
        records, total = store.get_history(page=1, page_size=3)
        assert total == 5
        assert len(records) == 3

    def test_outcome_update(self, store, sample_signal):
        store.publish(sample_signal)
        ok = store.update_outcome("sig-001", "WIN", 19.5)
        assert ok is True
        cur = store.get_current()
        assert cur.outcome == "WIN"
        assert cur.pnl_pips == 19.5
        assert cur.closed_at is not None

    def test_outcome_update_unknown_id(self, store):
        assert store.update_outcome("nope", "LOSS", -10) is False

    def test_thread_safety(self, store):
        """Concurrent publishes must not corrupt state."""
        errors = []

        def _writer(idx: int):
            try:
                for j in range(20):
                    store.publish(
                        SignalRecord(
                            signal_id=f"t{idx}-{j}",
                            action="HOLD",
                            symbol="XAUUSD",
                            entry_price=2350,
                            stop_loss=2340,
                            take_profit=2360,
                            rr_ratio=1.0,
                            created_at=datetime.now().isoformat(),
                        )
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        _, total = store.get_history(page=1, page_size=1)
        assert total == 80  # 4 threads * 20


# =============================================================================
# SIGNAL ENDPOINT TESTS
# =============================================================================

class TestSignalEndpoints:
    def test_no_signal_returns_204(self, client, auth_headers):
        resp = client.get("/api/v1/signals/current", headers=auth_headers)
        assert resp.status_code == 204

    def test_publish_then_200(self, client, sample_signal, auth_headers):
        store = client.app.state.app_state.signal_store
        store.publish(sample_signal)
        resp = client.get("/api/v1/signals/current", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["signal_id"] == "sig-001"
        assert body["action"] == "OPEN_LONG"
        assert body["symbol"] == "XAUUSD"
        assert body["rr_ratio"] == pytest.approx(1.86)

    def test_no_internal_risk_fields_leaked(self, client, sample_signal, auth_headers):
        store = client.app.state.app_state.signal_store
        store.publish(sample_signal)
        body = client.get("/api/v1/signals/current", headers=auth_headers).json()
        for forbidden in ("var", "sharpe", "garch", "drawdown", "correlation"):
            assert forbidden not in str(body).lower(), f"Leaked '{forbidden}'"

    def test_history_pagination(self, client, auth_headers):
        store = client.app.state.app_state.signal_store
        for i in range(25):
            store.publish(
                SignalRecord(
                    signal_id=f"h-{i:03d}",
                    action="HOLD",
                    symbol="XAUUSD",
                    entry_price=2350,
                    stop_loss=2340,
                    take_profit=2360,
                    rr_ratio=1.0,
                    created_at=f"2025-06-01T00:00:{i:02d}",
                )
            )
        resp = client.get("/api/v1/signals/history?page=2&page_size=10", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 25
        assert body["page"] == 2
        assert len(body["signals"]) == 10

    def test_history_outcome_fields(self, client, sample_signal, auth_headers):
        store = client.app.state.app_state.signal_store
        store.publish(sample_signal)
        store.update_outcome("sig-001", "WIN", 19.5)
        body = client.get("/api/v1/signals/history", headers=auth_headers).json()
        sig = body["signals"][0]
        assert sig["outcome"] == "WIN"
        assert sig["pnl_pips"] == pytest.approx(19.5)

    def test_page_size_validation(self, client, auth_headers):
        resp = client.get("/api/v1/signals/history?page_size=200", headers=auth_headers)
        assert resp.status_code == 422  # Pydantic validation


# =============================================================================
# HEALTH ENDPOINT TESTS
# =============================================================================

class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "uptime_seconds" in body
        assert body["is_trading_active"] is True

    def test_docker_health_compat(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body

    def test_health_with_kill_switch(self, client_with_kill_switch):
        resp = client_with_kill_switch.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["kill_switch_level"] == 0
        assert body["is_trading_active"] is True


# =============================================================================
# OPERATOR ENDPOINT TESTS
# =============================================================================

class TestOperatorEndpoints:
    def test_metrics_returns_200(self, client, auth_headers):
        resp = client.get("/api/v1/operator/metrics", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "metrics" in body

    def test_risk_returns_200(self, client, auth_headers):
        resp = client.get("/api/v1/operator/risk", headers=auth_headers)
        assert resp.status_code == 200

    def test_kill_switch_none_returns_empty(self, client, auth_headers):
        resp = client.get("/api/v1/operator/kill-switch", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["kill_switch"] == {}

    def test_kill_switch_with_mock(self, client_with_kill_switch, auth_headers):
        resp = client_with_kill_switch.get("/api/v1/operator/kill-switch", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["kill_switch"]["halt_level"] == "NONE"

    def test_risk_with_kill_switch(self, client_with_kill_switch, auth_headers):
        resp = client_with_kill_switch.get("/api/v1/operator/risk", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["current_drawdown_pct"] == pytest.approx(0.047)
        assert body["daily_pnl"] == pytest.approx(-200)

    def test_metrics_safe_defaults_when_none(self, tmp_path):
        """When no metrics_registry is provided, return empty dict."""
        ks = KeyStore(db_path=str(tmp_path / "ks.db"))
        api_key = ks.create_key("test")["api_key"]
        app = create_app(metrics_registry=None, key_store=ks)
        c = TestClient(app)
        resp = c.get("/api/v1/operator/metrics", headers={"X-API-Key": api_key})
        assert resp.status_code == 200
        assert resp.json()["metrics"] == {}


# =============================================================================
# PROMETHEUS ENDPOINT TESTS
# =============================================================================

class TestPrometheusEndpoint:
    def test_metrics_returns_text(self, client, registry):
        registry.counter("requests_total", "test").inc()
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        assert "requests_total" in resp.text

    def test_metrics_empty_when_no_registry(self):
        app = create_app(metrics_registry=None)
        c = TestClient(app)
        resp = c.get("/metrics")
        assert resp.status_code == 200
        assert resp.text == ""


# =============================================================================
# MIDDLEWARE TESTS
# =============================================================================

class TestMiddleware:
    def test_cors_headers_on_allowed_origin(self, client):
        resp = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_404_on_unknown_route(self, client):
        resp = client.get("/api/v1/nonexistent")
        assert resp.status_code == 404

    def test_cors_disallowed_origin(self, client):
        resp = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") is None

    def test_request_logging_records_histogram(self, client, registry):
        client.get("/api/v1/health")
        hist = registry.histogram("http_request_duration_seconds", "")
        summary = hist.get_summary(labels={"path": "/api/v1/health"})
        assert summary["count"] >= 1


# =============================================================================
# SIGNAL STORE EDGE CASE TESTS
# =============================================================================

class TestSignalStoreEdgeCases:
    def test_get_history_empty(self, store):
        records, total = store.get_history()
        assert records == []
        assert total == 0

    def test_pagination_beyond_total(self, store, sample_signal):
        store.publish(sample_signal)
        records, total = store.get_history(page=99, page_size=20)
        assert total == 1
        assert records == []

    def test_publish_overwrites_current(self, store, sample_signal):
        store.publish(sample_signal)
        second = SignalRecord(
            signal_id="sig-002",
            action="CLOSE_LONG",
            symbol="XAUUSD",
            entry_price=2370.0,
            stop_loss=2380.0,
            take_profit=2350.0,
            rr_ratio=2.0,
            created_at=datetime.now().isoformat(),
        )
        store.publish(second)
        cur = store.get_current()
        assert cur.signal_id == "sig-002"

    def test_history_ordered_descending(self, store):
        for i in range(3):
            store.publish(
                SignalRecord(
                    signal_id=f"ord-{i}",
                    action="HOLD",
                    symbol="XAUUSD",
                    entry_price=2350,
                    stop_loss=2340,
                    take_profit=2360,
                    rr_ratio=1.0,
                    created_at=f"2025-06-0{i+1}T12:00:00",
                )
            )
        records, _ = store.get_history()
        assert records[0].signal_id == "ord-2"  # newest first


# =============================================================================
# APP FACTORY TESTS
# =============================================================================

class TestAppFactory:
    def test_openapi_docs_accessible(self, client):
        resp = client.get("/api/docs")
        assert resp.status_code == 200

    def test_openapi_schema_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "Trading Bot Signal API"
        assert schema["info"]["version"] == "0.11.0"
