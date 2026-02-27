"""Shared test fixtures for the TradingBOT test suite."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from src.api.auth import KeyStore
from src.api.signal_store import SignalRecord, SignalStore
from src.api.signal_tracker import SignalTracker
from src.performance.metrics import MetricsRegistry


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture()
def tmp_db(tmp_path):
    """Temporary SQLite path for signal store."""
    return str(tmp_path / "test_signals.db")


@pytest.fixture()
def tmp_keys_db(tmp_path):
    """Temporary SQLite path for API key store."""
    return str(tmp_path / "test_api_keys.db")


@pytest.fixture()
def store(tmp_db):
    """Fresh SignalStore backed by temp DB."""
    return SignalStore(db_path=tmp_db)


@pytest.fixture()
def tracker(tmp_db):
    """Fresh SignalTracker reading from same temp DB as store."""
    return SignalTracker(db_path=tmp_db)


@pytest.fixture()
def key_store(tmp_keys_db):
    """Fresh KeyStore backed by temp DB."""
    return KeyStore(db_path=tmp_keys_db)


# =============================================================================
# AUTH FIXTURES
# =============================================================================

@pytest.fixture()
def test_api_key(key_store):
    """Pre-create a test subscriber key and return the raw key."""
    return key_store.create_key("test-subscriber")["api_key"]


@pytest.fixture()
def auth_headers(test_api_key):
    """Standard auth headers dict."""
    return {"X-API-Key": test_api_key}


# =============================================================================
# METRICS FIXTURE
# =============================================================================

@pytest.fixture()
def registry():
    """Fresh MetricsRegistry (test prefix to avoid collision)."""
    return MetricsRegistry(prefix="test")


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================

@pytest.fixture()
def sample_signal() -> SignalRecord:
    """Single sample signal record."""
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
def populated_store(store):
    """Store with 5 mixed win/loss signals, all closed recently."""
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
# MOCK FIXTURES
# =============================================================================

@pytest.fixture()
def mock_kill_switch():
    """Mock KillSwitch with safe defaults."""
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
    return ks
