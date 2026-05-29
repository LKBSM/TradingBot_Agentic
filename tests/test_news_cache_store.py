"""Tests for NewsCacheStore (Chantier 3)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.storage import NewsCacheStore, NormalizedNewsEvent


@pytest.fixture
def store(tmp_path):
    return NewsCacheStore(db_path=str(tmp_path / "news_cache.db"))


def _event(
    event_id: str = "abc123",
    currency: str = "USD",
    impact: str = "high",
    scheduled_at: datetime | None = None,
    actual=None,
    forecast=None,
) -> NormalizedNewsEvent:
    return NormalizedNewsEvent(
        event_id=event_id,
        event="US Non-Farm Payrolls",
        currency=currency,
        impact=impact,
        scheduled_at=scheduled_at or datetime(2026, 5, 29, 14, 30, tzinfo=timezone.utc),
        actual=actual,
        forecast=forecast,
    )


class TestRoundtrip:
    def test_upsert_then_query_window(self, store):
        ts = datetime(2026, 5, 29, 14, 30, tzinfo=timezone.utc)
        store.upsert_events([_event(scheduled_at=ts)])
        got = store.get_events_between(
            ts - timedelta(hours=1), ts + timedelta(hours=1)
        )
        assert len(got) == 1
        assert got[0].event == "US Non-Farm Payrolls"
        assert got[0].scheduled_at == ts

    def test_query_outside_window_returns_empty(self, store):
        ts = datetime(2026, 5, 29, 14, 30, tzinfo=timezone.utc)
        store.upsert_events([_event(scheduled_at=ts)])
        got = store.get_events_between(
            ts + timedelta(hours=2), ts + timedelta(hours=3)
        )
        assert got == []

    def test_empty_upsert_returns_zero(self, store):
        assert store.upsert_events([]) == 0


class TestDedup:
    def test_same_event_id_upserts_in_place(self, store):
        ts = datetime(2026, 5, 29, 14, 30, tzinfo=timezone.utc)
        store.upsert_events([_event(event_id="dup", actual=None)])
        # Re-fetch later with the actual value released — same id replaces row.
        store.upsert_events([_event(event_id="dup", actual=180000.0)])
        got = store.get_events_between(
            ts - timedelta(hours=1), ts + timedelta(hours=1)
        )
        assert len(got) == 1
        assert got[0].actual == 180000.0


class TestLastFetchAt:
    def test_none_when_empty(self, store):
        assert store.last_fetch_at() is None

    def test_returns_latest_fetched_at(self, store):
        t1 = datetime(2026, 5, 29, 14, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 5, 29, 14, 5, tzinfo=timezone.utc)
        store.upsert_events([_event(event_id="a")], fetched_at=t1)
        store.upsert_events([_event(event_id="b")], fetched_at=t2)
        last = store.last_fetch_at()
        assert last == t2


class TestPathResolution:
    def test_env_var_path(self, tmp_path, monkeypatch):
        target = tmp_path / "from_env.db"
        monkeypatch.setenv("NEWS_CACHE_DB_PATH", str(target))
        NewsCacheStore()
        assert target.exists()

    def test_explicit_path_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEWS_CACHE_DB_PATH", str(tmp_path / "env.db"))
        explicit = tmp_path / "explicit.db"
        NewsCacheStore(db_path=str(explicit))
        assert explicit.exists()
