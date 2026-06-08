"""Tests for MarketReadingsStore (Chantier 1)."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pytest

from src.storage import MarketReadingsStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "market_readings.db"
    return MarketReadingsStore(db_path=str(db_path))


def _sample_payload(close: float = 2378.45) -> Dict[str, Any]:
    """A minimal MarketReading payload matching Section 2.3 of the architecture doc."""
    return {
        "schema_version": "2.0.0",
        "header": {
            "instrument": "XAUUSD",
            "timeframe": "M15",
            "candle_close_ts": "2026-05-29T14:00:00Z",
            "close_price": close,
        },
        "structure": {
            "bos": {"direction": "bullish", "level": 2375.20},
            "order_blocks": [],
            "fair_value_gaps": [],
        },
        "regime": {"trend": "bullish", "volatility_observed": "elevated"},
        "events": {"news_upcoming": [], "news_just_published": []},
        "conditions": {"tags": ["volatility_elevated"], "description": "..."},
    }


# =============================================================================
# Roundtrip
# =============================================================================

class TestRoundtrip:
    def test_save_then_get_latest_returns_payload(self, store):
        ts = datetime(2026, 5, 29, 14, 0, tzinfo=timezone.utc)
        store.save_reading("XAUUSD", "M15", ts, _sample_payload())
        result = store.get_latest_reading("XAUUSD", "M15")
        assert result is not None
        assert result["header"]["close_price"] == 2378.45

    def test_get_latest_missing_returns_none(self, store):
        assert store.get_latest_reading("XAUUSD", "M15") is None

    def test_save_returns_positive_rowid(self, store):
        ts = datetime(2026, 5, 29, 14, 0, tzinfo=timezone.utc)
        rowid = store.save_reading("XAUUSD", "M15", ts, _sample_payload())
        assert isinstance(rowid, int)
        assert rowid > 0


# =============================================================================
# Idempotence (UPSERT)
# =============================================================================

class TestIdempotence:
    def test_save_same_key_replaces_payload(self, store):
        ts = datetime(2026, 5, 29, 14, 0, tzinfo=timezone.utc)
        store.save_reading("XAUUSD", "M15", ts, _sample_payload(close=100.0))
        store.save_reading("XAUUSD", "M15", ts, _sample_payload(close=200.0))
        latest = store.get_latest_reading("XAUUSD", "M15")
        assert latest["header"]["close_price"] == 200.0

    def test_save_same_key_does_not_create_duplicate_rows(self, store):
        ts = datetime(2026, 5, 29, 14, 0, tzinfo=timezone.utc)
        for _ in range(5):
            store.save_reading("XAUUSD", "M15", ts, _sample_payload())
        # Only one row should exist for that key
        history = store.get_readings_history(
            "XAUUSD", "M15",
            since=datetime(2026, 1, 1, tzinfo=timezone.utc),
            limit=100,
        )
        assert len(history) == 1

    def test_save_different_ts_keeps_both_rows(self, store):
        ts1 = datetime(2026, 5, 29, 14, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 5, 29, 14, 15, tzinfo=timezone.utc)
        store.save_reading("XAUUSD", "M15", ts1, _sample_payload(close=100.0))
        store.save_reading("XAUUSD", "M15", ts2, _sample_payload(close=200.0))
        history = store.get_readings_history(
            "XAUUSD", "M15", since=ts1, limit=10,
        )
        assert len(history) == 2


# =============================================================================
# History
# =============================================================================

class TestHistory:
    def test_filter_by_since(self, store):
        ts_old = datetime(2026, 5, 28, tzinfo=timezone.utc)
        ts_new = datetime(2026, 5, 29, tzinfo=timezone.utc)
        store.save_reading("XAUUSD", "M15", ts_old, _sample_payload(close=1.0))
        store.save_reading("XAUUSD", "M15", ts_new, _sample_payload(close=2.0))
        cutoff = datetime(2026, 5, 28, 12, tzinfo=timezone.utc)
        history = store.get_readings_history("XAUUSD", "M15", since=cutoff)
        assert len(history) == 1
        assert history[0]["header"]["close_price"] == 2.0

    def test_limit_applied(self, store):
        for i in range(20):
            ts = datetime(2026, 5, 29, 14, i, tzinfo=timezone.utc)
            store.save_reading("XAUUSD", "M15", ts, _sample_payload(close=2000.0 + i))
        history = store.get_readings_history(
            "XAUUSD", "M15",
            since=datetime(2026, 5, 1, tzinfo=timezone.utc),
            limit=5,
        )
        assert len(history) == 5

    def test_isolated_per_instrument_and_timeframe(self, store):
        ts = datetime(2026, 5, 29, tzinfo=timezone.utc)
        store.save_reading("XAUUSD", "M15", ts, _sample_payload(close=2000.0))
        store.save_reading("EURUSD", "M15", ts, _sample_payload(close=1.08))
        store.save_reading("XAUUSD", "H1", ts, _sample_payload(close=2050.0))
        hist_xau_m15 = store.get_readings_history("XAUUSD", "M15", since=ts)
        hist_eur_m15 = store.get_readings_history("EURUSD", "M15", since=ts)
        hist_xau_h1 = store.get_readings_history("XAUUSD", "H1", since=ts)
        assert len(hist_xau_m15) == 1 and hist_xau_m15[0]["header"]["close_price"] == 2000.0
        assert len(hist_eur_m15) == 1 and hist_eur_m15[0]["header"]["close_price"] == 1.08
        assert len(hist_xau_h1) == 1 and hist_xau_h1[0]["header"]["close_price"] == 2050.0


# =============================================================================
# Active combinations
# =============================================================================

class TestActiveCombinations:
    def test_mark_first_time(self, store):
        store.mark_combination_active("XAUUSD", "M15")
        combos = store.get_active_combinations(
            since=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        assert ("XAUUSD", "M15") in combos

    def test_mark_twice_keeps_single_row(self, store):
        store.mark_combination_active("XAUUSD", "M15")
        store.mark_combination_active("XAUUSD", "M15")
        combos = store.get_active_combinations(
            since=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        assert combos.count(("XAUUSD", "M15")) == 1

    def test_filter_by_since_future_excludes(self, store):
        store.mark_combination_active("XAUUSD", "M15")
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        combos = store.get_active_combinations(since=future)
        assert combos == []

    def test_multiple_combos_returned(self, store):
        store.mark_combination_active("XAUUSD", "M15")
        store.mark_combination_active("XAUUSD", "H1")
        store.mark_combination_active("EURUSD", "M15")
        combos = store.get_active_combinations(
            since=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        assert set(combos) == {
            ("XAUUSD", "M15"),
            ("XAUUSD", "H1"),
            ("EURUSD", "M15"),
        }


# =============================================================================
# Purge
# =============================================================================

class TestPurge:
    def test_purge_removes_old_only(self, store):
        # Insert a row, then backdate created_at directly via SQL to simulate "30 days ago"
        ts_old = datetime(2026, 5, 1, tzinfo=timezone.utc)
        store.save_reading("XAUUSD", "M15", ts_old, _sample_payload())
        conn = sqlite3.connect(str(store._db_path))
        try:
            conn.execute(
                "UPDATE market_readings SET created_at = datetime('now', '-30 days')"
            )
            conn.commit()
        finally:
            conn.close()

        # Insert a fresh row (created_at defaults to now)
        ts_new = datetime(2026, 5, 29, tzinfo=timezone.utc)
        store.save_reading("XAUUSD", "H1", ts_new, _sample_payload())

        deleted = store.purge_old_readings(older_than_days=7)
        assert deleted == 1

        # Only the fresh row should remain
        remaining = store.get_readings_history(
            "XAUUSD", "H1",
            since=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        assert len(remaining) == 1

    def test_purge_zero_when_no_old_rows(self, store):
        ts = datetime(2026, 5, 29, tzinfo=timezone.utc)
        store.save_reading("XAUUSD", "M15", ts, _sample_payload())
        deleted = store.purge_old_readings(older_than_days=7)
        assert deleted == 0


# =============================================================================
# JSON serialization (full Section 2.3 schema)
# =============================================================================

class TestJsonRoundtrip:
    def test_full_payload_roundtrip_deep_equal(self, store):
        ts = datetime(2026, 5, 29, 14, 0, tzinfo=timezone.utc)
        payload = _sample_payload()
        store.save_reading("XAUUSD", "M15", ts, payload)
        latest = store.get_latest_reading("XAUUSD", "M15")
        assert latest == payload

    def test_unicode_in_description_preserved(self, store):
        ts = datetime(2026, 5, 29, tzinfo=timezone.utc)
        payload = _sample_payload()
        payload["conditions"]["description"] = "Volatilité élevée — phase d'expansion"
        store.save_reading("XAUUSD", "M15", ts, payload)
        latest = store.get_latest_reading("XAUUSD", "M15")
        assert latest["conditions"]["description"] == "Volatilité élevée — phase d'expansion"

    def test_nested_lists_and_floats_preserved(self, store):
        ts = datetime(2026, 5, 29, tzinfo=timezone.utc)
        payload = _sample_payload()
        payload["structure"]["order_blocks"] = [
            {"id": "OB_001", "level_high": 2370.5, "level_low": 2368.2, "tested": False},
            {"id": "OB_002", "level_high": 2380.0, "level_low": 2378.0, "tested": True},
        ]
        store.save_reading("XAUUSD", "M15", ts, payload)
        latest = store.get_latest_reading("XAUUSD", "M15")
        assert latest["structure"]["order_blocks"] == payload["structure"]["order_blocks"]


# =============================================================================
# Thread-safety
# =============================================================================

class TestThreadSafety:
    def test_concurrent_writes_no_corruption(self, store):
        """2 threads × 50 writes each on disjoint keys → zero error, exact row count."""
        errors: list[Exception] = []
        N_PER_THREAD = 50

        def writer(thread_id: int) -> None:
            try:
                for i in range(N_PER_THREAD):
                    ts = datetime(2026, 5, 29, 14, 0, i, tzinfo=timezone.utc)
                    instrument = f"SYM_{thread_id}"
                    store.save_reading(
                        instrument, "M15", ts, _sample_payload(close=float(i)),
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent writers raised: {errors}"
        for thread_id in range(2):
            instrument = f"SYM_{thread_id}"
            history = store.get_readings_history(
                instrument, "M15",
                since=datetime(2026, 1, 1, tzinfo=timezone.utc),
                limit=N_PER_THREAD + 10,
            )
            assert len(history) == N_PER_THREAD


# =============================================================================
# Env-aware path resolution (Étape 4)
# =============================================================================

from pathlib import Path  # noqa: E402  (intentional late import — only used here)


class TestEnvAwarePath:
    def test_explicit_path_wins_over_env_var(self, tmp_path, monkeypatch):
        env_path = tmp_path / "from_env.db"
        explicit_path = tmp_path / "explicit.db"
        monkeypatch.setenv("MARKET_READINGS_DB_PATH", str(env_path))
        store = MarketReadingsStore(db_path=str(explicit_path))
        assert store._db_path == explicit_path

    def test_env_var_used_when_no_explicit_path(self, tmp_path, monkeypatch):
        env_path = tmp_path / "from_env.db"
        monkeypatch.setenv("MARKET_READINGS_DB_PATH", str(env_path))
        store = MarketReadingsStore()
        assert store._db_path == env_path

    def test_fallback_to_default_when_no_arg_no_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MARKET_READINGS_DB_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        store = MarketReadingsStore()
        assert store._db_path == Path("./data/market_readings.db")

    def test_empty_env_var_treated_as_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MARKET_READINGS_DB_PATH", "")
        monkeypatch.chdir(tmp_path)
        store = MarketReadingsStore()
        assert store._db_path == Path("./data/market_readings.db")
