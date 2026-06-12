"""Tests for CandlesCacheStore (Chantier 1 — Étape 5)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.intelligence.data_providers.twelve_data_provider import Candle
from src.storage import CandlesCacheStore


@pytest.fixture
def store(tmp_path):
    return CandlesCacheStore(db_path=str(tmp_path / "candles.db"))


def _sample_candle(seconds: int = 0, close: float = 2378.0) -> Candle:
    return Candle(
        ts=datetime(2026, 5, 29, 14, 0, seconds, tzinfo=timezone.utc),
        open=2377.0, high=2380.0, low=2376.0, close=close, volume=100.0,
    )


class TestRoundtrip:
    def test_upsert_then_get_latest(self, store):
        store.upsert_candles("XAUUSD", "M15", [_sample_candle(close=2378.7)])
        latest = store.get_latest_candle("XAUUSD", "M15")
        assert latest is not None
        assert latest.close == 2378.7

    def test_count_candles(self, store):
        candles = [_sample_candle(seconds=i, close=2378.0 + i) for i in range(5)]
        store.upsert_candles("XAUUSD", "M15", candles)
        assert store.count_candles("XAUUSD", "M15") == 5

    def test_get_latest_missing_returns_none(self, store):
        assert store.get_latest_candle("XAUUSD", "M15") is None

    def test_count_zero_when_empty(self, store):
        assert store.count_candles("XAUUSD", "M15") == 0


class TestIdempotence:
    def test_upsert_same_ts_replaces(self, store):
        c1 = _sample_candle(close=100.0)
        c2 = _sample_candle(close=200.0)  # same ts, different close
        store.upsert_candles("XAUUSD", "M15", [c1])
        store.upsert_candles("XAUUSD", "M15", [c2])
        assert store.get_latest_candle("XAUUSD", "M15").close == 200.0
        assert store.count_candles("XAUUSD", "M15") == 1

    def test_upsert_empty_returns_zero(self, store):
        assert store.upsert_candles("XAUUSD", "M15", []) == 0

    def test_isolated_per_instrument_and_timeframe(self, store):
        c = _sample_candle()
        store.upsert_candles("XAUUSD", "M15", [c])
        store.upsert_candles("EURUSD", "M15", [c])
        store.upsert_candles("XAUUSD", "H1", [c])
        assert store.count_candles("XAUUSD", "M15") == 1
        assert store.count_candles("EURUSD", "M15") == 1
        assert store.count_candles("XAUUSD", "H1") == 1


class TestGetLastNCandles:
    def test_returns_ascending_order(self, store):
        # Insert newest-to-oldest to prove the store sorts, not the caller.
        candles = [_sample_candle(seconds=i, close=2378.0 + i) for i in range(5)]
        store.upsert_candles("XAUUSD", "M15", list(reversed(candles)))
        got = store.get_last_n_candles("XAUUSD", "M15", 5)
        assert [c.close for c in got] == [2378.0 + i for i in range(5)]
        # Strictly ascending timestamps.
        assert all(got[i].ts < got[i + 1].ts for i in range(len(got) - 1))

    def test_limit_returns_most_recent_window(self, store):
        candles = [_sample_candle(seconds=i, close=2378.0 + i) for i in range(10)]
        store.upsert_candles("XAUUSD", "M15", candles)
        got = store.get_last_n_candles("XAUUSD", "M15", 3)
        assert len(got) == 3
        # The 3 most recent (highest seconds), still ascending.
        assert [c.close for c in got] == [2385.0, 2386.0, 2387.0]

    def test_empty_when_no_candles(self, store):
        assert store.get_last_n_candles("XAUUSD", "M15", 50) == []

    def test_zero_or_negative_limit_returns_empty(self, store):
        store.upsert_candles("XAUUSD", "M15", [_sample_candle()])
        assert store.get_last_n_candles("XAUUSD", "M15", 0) == []
        assert store.get_last_n_candles("XAUUSD", "M15", -3) == []

    def test_isolated_per_combo(self, store):
        store.upsert_candles("XAUUSD", "M15", [_sample_candle(close=1.0)])
        store.upsert_candles("EURUSD", "M15", [_sample_candle(close=2.0)])
        xau = store.get_last_n_candles("XAUUSD", "M15", 10)
        assert [c.close for c in xau] == [1.0]


class TestEnvAwarePath:
    def test_explicit_path_wins_over_env_var(self, tmp_path, monkeypatch):
        env_path = tmp_path / "from_env.db"
        explicit_path = tmp_path / "explicit.db"
        monkeypatch.setenv("CANDLES_DB_PATH", str(env_path))
        store = CandlesCacheStore(db_path=str(explicit_path))
        assert store._db_path == explicit_path

    def test_env_var_used_when_no_explicit_path(self, tmp_path, monkeypatch):
        env_path = tmp_path / "from_env.db"
        monkeypatch.setenv("CANDLES_DB_PATH", str(env_path))
        store = CandlesCacheStore()
        assert store._db_path == env_path

    def test_fallback_to_default_when_no_arg_no_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CANDLES_DB_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        store = CandlesCacheStore()
        assert store._db_path == Path("./data/candles.db")

    def test_empty_env_var_treated_as_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CANDLES_DB_PATH", "")
        monkeypatch.chdir(tmp_path)
        store = CandlesCacheStore()
        assert store._db_path == Path("./data/candles.db")


class TestSchemaV2WipesPollutedV1Cache:
    """Audit 2026-06-12 §T2/T3: v1 rows carry exchange-local timestamps
    mislabelled as UTC plus forming-bar values. Upgrading to v2 wipes the
    cache once; the assembler refills it with closed UTC bars."""

    def test_v1_rows_are_wiped_on_upgrade(self, tmp_path):
        import sqlite3

        db_path = tmp_path / "candles_v1.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """
            CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
            INSERT INTO schema_version (version) VALUES (1);
            CREATE TABLE candles_cache (
                instrument TEXT NOT NULL, timeframe TEXT NOT NULL,
                ts TEXT NOT NULL, open REAL NOT NULL, high REAL NOT NULL,
                low REAL NOT NULL, close REAL NOT NULL, volume REAL,
                PRIMARY KEY(instrument, timeframe, ts)
            );
            INSERT INTO candles_cache VALUES
                ('XAUUSD','M15','2026-06-13T00:00:00+00:00',4186.9,4197.4,4184.9,4192.2,0.0);
            """
        )
        conn.commit()
        conn.close()

        store = CandlesCacheStore(db_path=str(db_path))
        assert store.count_candles("XAUUSD", "M15") == 0

    def test_fresh_db_starts_at_v2_without_wipe_side_effects(self, tmp_path):
        store = CandlesCacheStore(db_path=str(tmp_path / "fresh.db"))
        c = Candle(
            ts=datetime(2026, 6, 12, 14, 0, tzinfo=timezone.utc),
            open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0,
        )
        store.upsert_candles("XAUUSD", "M15", [c])
        assert store.count_candles("XAUUSD", "M15") == 1
