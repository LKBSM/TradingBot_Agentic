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
