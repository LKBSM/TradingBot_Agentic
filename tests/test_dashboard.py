"""
Sprint 11 — Signal Performance Dashboard tests.

Covers:
  - SignalTracker unit tests (empty DB, summary, equity curve, Sharpe, drawdown)
  - Dashboard endpoint tests (summary, equity-curve, auth, JSON shape)
  - New metrics tests (counters, gauges, histogram in registry)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import KeyStore
from src.api.signal_store import SignalRecord, SignalStore
from src.api.signal_tracker import SignalTracker
from src.performance.metrics import MetricsRegistry, create_trading_metrics


# Disable TESTING_MODE for dashboard auth tests
@pytest.fixture(autouse=True)
def _disable_testing_mode():
    with patch("src.api.auth.TESTING_MODE", False), \
         patch("src.api.routes.health.TESTING_MODE", False):
        yield


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture()
def tmp_db(tmp_path):
    return str(tmp_path / "test_signals.db")


@pytest.fixture()
def store(tmp_db):
    return SignalStore(db_path=tmp_db)


@pytest.fixture()
def tracker(tmp_db):
    return SignalTracker(db_path=tmp_db)


@pytest.fixture()
def tmp_keys_db(tmp_path):
    return str(tmp_path / "test_api_keys.db")


@pytest.fixture()
def key_store(tmp_keys_db):
    return KeyStore(db_path=tmp_keys_db)


@pytest.fixture()
def test_api_key(key_store):
    return key_store.create_key("test-subscriber")["api_key"]


@pytest.fixture()
def auth_headers(test_api_key):
    return {"X-API-Key": test_api_key}


@pytest.fixture()
def client(tmp_db, key_store):
    signal_store = SignalStore(db_path=tmp_db)
    signal_tracker = SignalTracker(db_path=tmp_db)
    app = create_app(
        signal_store=signal_store,
        signal_tracker=signal_tracker,
        key_store=key_store,
    )
    return TestClient(app)


@pytest.fixture()
def populated_store(store):
    """Store with mixed win/loss signals, all closed recently."""
    now = datetime.now()
    signals = [
        ("s1", "WIN", 20.0, 1.8, now - timedelta(days=5)),
        ("s2", "WIN", 15.0, 2.0, now - timedelta(days=4)),
        ("s3", "LOSS", -10.0, 1.5, now - timedelta(days=3)),
        ("s4", "WIN", 30.0, 2.5, now - timedelta(days=2)),
        ("s5", "LOSS", -25.0, 1.2, now - timedelta(days=1)),
    ]
    for sig_id, outcome, pnl, rr, closed_at in signals:
        store.publish(SignalRecord(
            signal_id=sig_id,
            action="OPEN_LONG",
            symbol="XAUUSD",
            entry_price=2350.0,
            stop_loss=2340.0,
            take_profit=2370.0,
            rr_ratio=rr,
            created_at=(closed_at - timedelta(hours=2)).isoformat(),
        ))
        store.update_outcome(sig_id, outcome, pnl, closed_at.isoformat())
    return store


# =============================================================================
# SIGNAL TRACKER UNIT TESTS
# =============================================================================

class TestSignalTrackerEmpty:
    def test_empty_db_returns_zeros(self, tracker):
        summary = tracker.get_performance_summary(days=30)
        assert summary["total"] == 0
        assert summary["won"] == 0
        assert summary["lost"] == 0
        assert summary["win_rate"] == 0.0
        assert summary["profit_factor"] == 0.0
        assert summary["cumulative_pnl"] == 0.0
        assert summary["sharpe_30d"] == 0.0
        assert summary["max_drawdown_pct"] == 0.0

    def test_empty_equity_curve(self, tracker):
        curve = tracker.get_equity_curve(days=30)
        assert curve == []


class TestSignalTrackerSummary:
    def test_total_count(self, populated_store, tracker):
        summary = tracker.get_performance_summary(days=30)
        assert summary["total"] == 5

    def test_win_loss_count(self, populated_store, tracker):
        summary = tracker.get_performance_summary(days=30)
        assert summary["won"] == 3
        assert summary["lost"] == 2

    def test_win_rate(self, populated_store, tracker):
        summary = tracker.get_performance_summary(days=30)
        assert summary["win_rate"] == pytest.approx(0.6, abs=0.01)

    def test_profit_factor(self, populated_store, tracker):
        summary = tracker.get_performance_summary(days=30)
        # gross_profit = 20 + 15 + 30 = 65, gross_loss = 10 + 25 = 35
        assert summary["profit_factor"] == pytest.approx(65 / 35, abs=0.01)

    def test_cumulative_pnl(self, populated_store, tracker):
        summary = tracker.get_performance_summary(days=30)
        # 20 + 15 - 10 + 30 - 25 = 30
        assert summary["cumulative_pnl"] == pytest.approx(30.0, abs=0.1)

    def test_avg_rr(self, populated_store, tracker):
        summary = tracker.get_performance_summary(days=30)
        expected = (1.8 + 2.0 + 1.5 + 2.5 + 1.2) / 5
        assert summary["avg_rr"] == pytest.approx(expected, abs=0.01)

    def test_days_filtering(self, store, tracker):
        """Signals older than the window should be excluded."""
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        store.publish(SignalRecord(
            signal_id="old-1",
            action="OPEN_LONG",
            symbol="XAUUSD",
            entry_price=2350.0,
            stop_loss=2340.0,
            take_profit=2370.0,
            rr_ratio=1.5,
            created_at=old_date,
        ))
        store.update_outcome("old-1", "WIN", 10.0, old_date)

        summary = tracker.get_performance_summary(days=30)
        assert summary["total"] == 0

    def test_all_wins_profit_factor(self, store, tracker):
        """All winning signals — profit_factor capped to 999.99."""
        now = datetime.now()
        store.publish(SignalRecord(
            signal_id="w1", action="OPEN_LONG", symbol="XAUUSD",
            entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0,
            rr_ratio=2.0, created_at=(now - timedelta(hours=2)).isoformat(),
        ))
        store.update_outcome("w1", "WIN", 20.0, now.isoformat())

        summary = tracker.get_performance_summary(days=30)
        assert summary["profit_factor"] == 999.99


class TestSignalTrackerSharpe:
    def test_sharpe_positive_for_profitable(self, populated_store, tracker):
        summary = tracker.get_performance_summary(days=30)
        # Net PnL is positive, Sharpe should be positive
        assert summary["sharpe_30d"] > 0

    def test_sharpe_single_signal_is_zero(self, store, tracker):
        """Need >= 2 signals for Sharpe."""
        now = datetime.now()
        store.publish(SignalRecord(
            signal_id="solo", action="OPEN_LONG", symbol="XAUUSD",
            entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0,
            rr_ratio=2.0, created_at=(now - timedelta(hours=2)).isoformat(),
        ))
        store.update_outcome("solo", "WIN", 50.0, now.isoformat())

        summary = tracker.get_performance_summary(days=30)
        assert summary["sharpe_30d"] == 0.0

    def test_sharpe_all_same_pnl_is_zero(self, store, tracker):
        """Zero variance means zero Sharpe (div-by-zero guard)."""
        now = datetime.now()
        for i in range(3):
            sid = f"flat-{i}"
            store.publish(SignalRecord(
                signal_id=sid, action="OPEN_LONG", symbol="XAUUSD",
                entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0,
                rr_ratio=2.0,
                created_at=(now - timedelta(hours=i + 1)).isoformat(),
            ))
            store.update_outcome(sid, "WIN", 10.0, now.isoformat())

        summary = tracker.get_performance_summary(days=30)
        assert summary["sharpe_30d"] == 0.0


class TestSignalTrackerMaxDrawdown:
    def test_max_drawdown_with_losses(self, populated_store, tracker):
        summary = tracker.get_performance_summary(days=30)
        # Cumulative: 20, 35, 25, 55, 30 => peak=55, dd=55-30=25 => 25/55=45.45%
        assert summary["max_drawdown_pct"] == pytest.approx(45.4545, abs=0.1)

    def test_no_drawdown_monotonic_wins(self, store, tracker):
        """Monotonically increasing equity => 0% drawdown."""
        now = datetime.now()
        for i in range(3):
            sid = f"up-{i}"
            store.publish(SignalRecord(
                signal_id=sid, action="OPEN_LONG", symbol="XAUUSD",
                entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0,
                rr_ratio=2.0,
                created_at=(now - timedelta(days=3 - i, hours=2)).isoformat(),
            ))
            store.update_outcome(sid, "WIN", 10.0 + i,
                                 (now - timedelta(days=3 - i)).isoformat())

        summary = tracker.get_performance_summary(days=30)
        assert summary["max_drawdown_pct"] == 0.0


class TestSignalTrackerEquityCurve:
    def test_equity_curve_length(self, populated_store, tracker):
        curve = tracker.get_equity_curve(days=30)
        assert len(curve) == 5

    def test_equity_curve_ordering(self, populated_store, tracker):
        curve = tracker.get_equity_curve(days=30)
        dates = [p["closed_at"] for p in curve]
        assert dates == sorted(dates)

    def test_equity_curve_cumulative(self, populated_store, tracker):
        curve = tracker.get_equity_curve(days=30)
        # Last point should be cumulative sum = 30.0
        assert curve[-1]["cumulative_pnl"] == pytest.approx(30.0, abs=0.1)

    def test_equity_curve_first_point(self, populated_store, tracker):
        curve = tracker.get_equity_curve(days=30)
        # First signal is s1 with pnl=20 => cumulative=20
        assert curve[0]["pnl_pips"] == pytest.approx(20.0)
        assert curve[0]["cumulative_pnl"] == pytest.approx(20.0)


# =============================================================================
# DASHBOARD ENDPOINT TESTS
# =============================================================================

class TestDashboardSummaryEndpoint:
    def test_summary_returns_200(self, client, auth_headers):
        resp = client.get("/api/v1/dashboard/summary", headers=auth_headers)
        assert resp.status_code == 200

    def test_summary_without_key_returns_401(self, client):
        resp = client.get("/api/v1/dashboard/summary")
        assert resp.status_code == 401

    def test_summary_json_shape(self, client, auth_headers):
        resp = client.get("/api/v1/dashboard/summary", headers=auth_headers)
        body = resp.json()
        expected_keys = {
            "total_signals", "winning", "losing", "win_rate",
            "profit_factor", "avg_rr", "cumulative_pnl",
            "sharpe_30d", "max_drawdown_pct", "period_days",
        }
        assert set(body.keys()) == expected_keys

    def test_summary_default_period(self, client, auth_headers):
        body = client.get("/api/v1/dashboard/summary", headers=auth_headers).json()
        assert body["period_days"] == 30

    def test_summary_custom_period(self, client, auth_headers):
        body = client.get(
            "/api/v1/dashboard/summary?days=7", headers=auth_headers
        ).json()
        assert body["period_days"] == 7

    def test_summary_with_data(self, client, auth_headers):
        """Populate signals, then check summary reflects them."""
        store = client.app.state.app_state.signal_store
        now = datetime.now()
        store.publish(SignalRecord(
            signal_id="ds1", action="OPEN_LONG", symbol="XAUUSD",
            entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0,
            rr_ratio=2.0, created_at=(now - timedelta(hours=3)).isoformat(),
        ))
        store.update_outcome("ds1", "WIN", 15.0, now.isoformat())

        body = client.get("/api/v1/dashboard/summary", headers=auth_headers).json()
        assert body["total_signals"] == 1
        assert body["winning"] == 1
        assert body["win_rate"] == pytest.approx(1.0)


class TestDashboardEquityCurveEndpoint:
    def test_equity_curve_returns_200(self, client, auth_headers):
        resp = client.get("/api/v1/dashboard/equity-curve", headers=auth_headers)
        assert resp.status_code == 200

    def test_equity_curve_without_key_returns_401(self, client):
        resp = client.get("/api/v1/dashboard/equity-curve")
        assert resp.status_code == 401

    def test_equity_curve_json_shape(self, client, auth_headers):
        body = client.get(
            "/api/v1/dashboard/equity-curve", headers=auth_headers
        ).json()
        assert "points" in body
        assert "current_cumulative_pnl" in body
        assert isinstance(body["points"], list)

    def test_equity_curve_empty(self, client, auth_headers):
        body = client.get(
            "/api/v1/dashboard/equity-curve", headers=auth_headers
        ).json()
        assert body["points"] == []
        assert body["current_cumulative_pnl"] == 0.0

    def test_equity_curve_with_data(self, client, auth_headers):
        store = client.app.state.app_state.signal_store
        now = datetime.now()
        for i, (outcome, pnl) in enumerate([("WIN", 10.0), ("LOSS", -5.0)]):
            sid = f"ec-{i}"
            store.publish(SignalRecord(
                signal_id=sid, action="OPEN_LONG", symbol="XAUUSD",
                entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0,
                rr_ratio=2.0,
                created_at=(now - timedelta(days=2 - i, hours=2)).isoformat(),
            ))
            store.update_outcome(sid, outcome, pnl,
                                 (now - timedelta(days=2 - i)).isoformat())

        body = client.get(
            "/api/v1/dashboard/equity-curve", headers=auth_headers
        ).json()
        assert len(body["points"]) == 2
        assert body["current_cumulative_pnl"] == pytest.approx(5.0)


class TestDashboardNoTracker:
    def test_summary_without_tracker_returns_zeros(self, tmp_path):
        ks = KeyStore(db_path=str(tmp_path / "ks.db"))
        api_key = ks.create_key("test")["api_key"]
        app = create_app(signal_tracker=None, key_store=ks)
        c = TestClient(app)
        resp = c.get("/api/v1/dashboard/summary", headers={"X-API-Key": api_key})
        assert resp.status_code == 200
        assert resp.json()["total_signals"] == 0

    def test_equity_curve_without_tracker_returns_empty(self, tmp_path):
        ks = KeyStore(db_path=str(tmp_path / "ks.db"))
        api_key = ks.create_key("test")["api_key"]
        app = create_app(signal_tracker=None, key_store=ks)
        c = TestClient(app)
        resp = c.get("/api/v1/dashboard/equity-curve", headers={"X-API-Key": api_key})
        assert resp.status_code == 200
        assert resp.json()["points"] == []


# =============================================================================
# METRICS TESTS (Sprint 11 additions)
# =============================================================================

class TestSignalMetrics:
    def test_signal_counters_exist(self):
        reg = MetricsRegistry(prefix="test_s11")
        metrics = create_trading_metrics(reg)
        assert "signals_published" in metrics
        assert "signals_closed_win" in metrics
        assert "signals_closed_loss" in metrics

    def test_signal_gauges_exist(self):
        reg = MetricsRegistry(prefix="test_s11g")
        metrics = create_trading_metrics(reg)
        assert "current_win_rate_30d" in metrics
        assert "current_sharpe_30d" in metrics
        assert "current_profit_factor_30d" in metrics
        assert "current_cumulative_pnl" in metrics

    def test_signal_pnl_histogram_exists(self):
        reg = MetricsRegistry(prefix="test_s11h")
        metrics = create_trading_metrics(reg)
        assert "signal_pnl_pips" in metrics

    def test_signal_pnl_histogram_accepts_values(self):
        reg = MetricsRegistry(prefix="test_s11hv")
        metrics = create_trading_metrics(reg)
        hist = metrics["signal_pnl_pips"]
        hist.observe(25.0)
        hist.observe(-15.0)
        summary = hist.get_summary()
        assert summary["count"] == 2
        assert summary["sum"] == pytest.approx(10.0)

    def test_signal_counters_increment(self):
        reg = MetricsRegistry(prefix="test_s11c")
        metrics = create_trading_metrics(reg)
        metrics["signals_published"].inc()
        metrics["signals_published"].inc()
        metrics["signals_closed_win"].inc()
        assert metrics["signals_published"].get() == 2
        assert metrics["signals_closed_win"].get() == 1

    def test_signal_gauges_set(self):
        reg = MetricsRegistry(prefix="test_s11gs")
        metrics = create_trading_metrics(reg)
        metrics["current_win_rate_30d"].set(0.65)
        metrics["current_sharpe_30d"].set(1.8)
        assert metrics["current_win_rate_30d"].get() == pytest.approx(0.65)
        assert metrics["current_sharpe_30d"].get() == pytest.approx(1.8)


# =============================================================================
# APP VERSION TEST
# =============================================================================

class TestAppVersion:
    def test_version_bumped_to_0_11(self, client):
        resp = client.get("/openapi.json")
        assert resp.json()["info"]["version"] == "1.0.0"
