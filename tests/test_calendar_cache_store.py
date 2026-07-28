"""CalendarCacheStore — persistence, revisions, all-impact range, coverage (NW-1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.storage.calendar_cache_store import CalendarCacheEvent, CalendarCacheStore


def _ev(event_id: str = "official:cpi-2026-07", **kw) -> CalendarCacheEvent:
    base = dict(
        event_id=event_id,
        source="official",
        event="CPI",
        currency="USD",
        impact="high",
        scheduled_at=datetime(2026, 7, 28, 12, 30, tzinfo=timezone.utc),
        markets=["XAUUSD", "EURUSD"],
    )
    base.update(kw)
    return CalendarCacheEvent(**base)


@pytest.fixture()
def store(tmp_path) -> CalendarCacheStore:
    return CalendarCacheStore(db_path=str(tmp_path / "cal.db"))


def test_roundtrip_preserves_neutral_fields(store: CalendarCacheStore) -> None:
    store.upsert_events(
        [_ev(series_code="CUSR0000SA0", organism="BLS", value_unit="% y/y", actual=3.1)]
    )
    got = store.get_events_between(
        datetime(2026, 7, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    assert len(got) == 1
    e = got[0]
    assert e.series_code == "CUSR0000SA0"
    assert e.organism == "BLS"
    assert e.value_unit == "% y/y"
    assert e.markets == ["XAUUSD", "EURUSD"]


def test_null_fields_stay_none_never_defaulted(store: CalendarCacheStore) -> None:
    # A prototype-adapter event: organism / series_code / unit absent.
    store.upsert_events([_ev(organism=None, series_code=None, value_unit=None)])
    e = store.get_events_between(
        datetime(2026, 7, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 1, tzinfo=timezone.utc),
    )[0]
    assert e.organism is None
    assert e.series_code is None
    assert e.value_unit is None


def test_revision_flags_and_keeps_pre_revision_value(store: CalendarCacheStore) -> None:
    store.upsert_events([_ev(actual=3.1)])
    # Same event_id, actual changes → revised, prior value retained.
    store.upsert_events([_ev(actual=3.3)])
    e = store.get_events_between(
        datetime(2026, 7, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 1, tzinfo=timezone.utc),
    )[0]
    assert e.revised is True
    assert e.actual == 3.3
    assert e.previous_before_revision == 3.1


def test_no_revision_when_values_unchanged(store: CalendarCacheStore) -> None:
    store.upsert_events([_ev(actual=3.1)])
    store.upsert_events([_ev(actual=3.1)])
    e = store.get_events_between(
        datetime(2026, 7, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 1, tzinfo=timezone.utc),
    )[0]
    assert e.revised is False
    assert e.previous_before_revision is None


def test_all_impacts_persist_including_low(store: CalendarCacheStore) -> None:
    for i, imp in enumerate(("high", "medium", "low")):
        store.upsert_events(
            [_ev(event_id=f"official:e{i}", impact=imp, event=f"E{i}")]
        )
    got = store.get_events_between(
        datetime(2026, 7, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    assert {e.impact for e in got} == {"high", "medium", "low"}


def test_coverage_bounds(store: CalendarCacheStore) -> None:
    assert store.coverage_bounds() == (None, None)
    lo = datetime(2026, 7, 20, tzinfo=timezone.utc)
    hi = datetime(2026, 7, 31, tzinfo=timezone.utc)
    store.upsert_events(
        [
            _ev(event_id="official:a", scheduled_at=lo),
            _ev(event_id="official:b", scheduled_at=hi),
        ]
    )
    assert store.coverage_bounds() == (lo, hi)


def test_two_sources_no_id_conflict(store: CalendarCacheStore) -> None:
    ts = datetime(2026, 7, 28, 12, 30, tzinfo=timezone.utc)
    store.upsert_events(
        [
            _ev(event_id="official:cpi", source="official", scheduled_at=ts),
            _ev(event_id="forexfactory:cpi", source="forexfactory", scheduled_at=ts),
        ]
    )
    got = store.get_events_between(
        ts - timedelta(days=1), ts + timedelta(days=1)
    )
    assert {e.source for e in got} == {"official", "forexfactory"}
    assert len(got) == 2
