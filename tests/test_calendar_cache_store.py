"""CalendarCacheStore (NW-1 / NW-1b) — persistence, revisions (initial vs
current + date), per-source freshness, coverage, no impact/forecast columns."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.storage.calendar_cache_store import CalendarCacheEvent, CalendarCacheStore


def _ev(event_id: str = "bls:cpi-2026-07", **kw) -> CalendarCacheEvent:
    base = dict(
        event_id=event_id,
        source="bls",
        event="CPI",
        currency="USD",
        scheduled_at=datetime(2026, 7, 28, 12, 30, tzinfo=timezone.utc),
        markets=["XAUUSD", "EURUSD"],
    )
    base.update(kw)
    return CalendarCacheEvent(**base)


WINDOW = (datetime(2026, 7, 1, tzinfo=timezone.utc),
          datetime(2026, 8, 1, tzinfo=timezone.utc))


@pytest.fixture()
def store(tmp_path) -> CalendarCacheStore:
    return CalendarCacheStore(db_path=str(tmp_path / "cal.db"))


def test_roundtrip_preserves_neutral_official_fields(store: CalendarCacheStore) -> None:
    store.upsert_events([_ev(
        series_code="CUUR0000SA0", organism="Bureau of Labor Statistics",
        value_unit="indice", periodicity="monthly", time_confirmed=True, actual=3.1,
        license_label="Domaine public",
    )])
    e = store.get_events_between(*WINDOW)[0]
    assert e.series_code == "CUUR0000SA0"
    assert e.organism == "Bureau of Labor Statistics"
    assert e.value_unit == "indice"
    assert e.periodicity == "monthly"
    assert e.time_confirmed is True
    assert e.license_label == "Domaine public"
    assert e.markets == ["XAUUSD", "EURUSD"]


def test_time_confirmed_persists_false(store: CalendarCacheStore) -> None:
    store.upsert_events([_ev(time_confirmed=False)])
    assert store.get_events_between(*WINDOW)[0].time_confirmed is False


def test_value_stored_as_published_no_rounding(store: CalendarCacheStore) -> None:
    store.upsert_events([_ev(actual=3.14159, previous=2.71828)])
    e = store.get_events_between(*WINDOW)[0]
    assert e.actual == 3.14159   # exact, never re-rounded
    assert e.previous == 2.71828


def test_revision_keeps_initial_and_dates_it(store: CalendarCacheStore) -> None:
    t0 = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 8, 27, 9, 0, tzinfo=timezone.utc)
    store.upsert_events([_ev(actual=3.1)], fetched_at=t0)
    store.upsert_events([_ev(actual=3.3)], fetched_at=t1)  # same id, actual changes
    e = store.get_events_between(*WINDOW)[0]
    assert e.revised is True
    assert e.actual == 3.3            # current value
    assert e.actual_initial == 3.1    # first-published value, never overwritten
    assert e.revised_at == t1         # dated


def test_never_revised_says_so(store: CalendarCacheStore) -> None:
    store.upsert_events([_ev(actual=3.1)])
    store.upsert_events([_ev(actual=3.1)])  # unchanged
    e = store.get_events_between(*WINDOW)[0]
    assert e.revised is False
    assert e.revised_at is None
    assert e.actual_initial == 3.1   # initial known, just never revised


def test_first_print_captured_as_initial(store: CalendarCacheStore) -> None:
    # A future event (no value) then its first publication → that IS the initial.
    store.upsert_events([_ev(actual=None)])
    store.upsert_events([_ev(actual=5.0)])
    e = store.get_events_between(*WINDOW)[0]
    assert e.actual == 5.0
    assert e.actual_initial == 5.0
    assert e.revised is False


def test_no_impact_or_forecast_column(store: CalendarCacheStore) -> None:
    import sqlite3
    conn = sqlite3.connect(store._db_path)  # type: ignore[attr-defined]
    cols = {r[1] for r in conn.execute("PRAGMA table_info(calendar_cache)")}
    conn.close()
    assert "impact" not in cols
    assert "forecast" not in cols
    assert {"periodicity", "time_confirmed", "actual_initial", "revised_at"} <= cols


def test_source_last_success_per_source(store: CalendarCacheStore) -> None:
    t0 = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 7, 28, 10, 0, tzinfo=timezone.utc)
    store.upsert_events([_ev(event_id="bls:a", source="bls")], fetched_at=t0)
    store.upsert_events([_ev(event_id="ecb:b", source="ecb")], fetched_at=t1)
    ls = store.source_last_success()
    assert ls["bls"] == t0
    assert ls["ecb"] == t1


def test_unavailable_source_rows_are_not_deleted(store: CalendarCacheStore) -> None:
    # Upserting an empty batch (source returned nothing) never deletes.
    store.upsert_events([_ev(actual=3.1)])
    store.upsert_events([])  # source unavailable this cycle
    assert len(store.get_events_between(*WINDOW)) == 1


def test_coverage_bounds(store: CalendarCacheStore) -> None:
    assert store.coverage_bounds() == (None, None)
    lo = datetime(2026, 7, 20, tzinfo=timezone.utc)
    hi = datetime(2026, 7, 31, tzinfo=timezone.utc)
    store.upsert_events([_ev(event_id="bls:a", scheduled_at=lo),
                         _ev(event_id="bls:b", scheduled_at=hi)])
    assert store.coverage_bounds() == (lo, hi)


def test_two_sources_no_id_conflict(store: CalendarCacheStore) -> None:
    ts = datetime(2026, 7, 28, 12, 30, tzinfo=timezone.utc)
    store.upsert_events([
        _ev(event_id="bls:cpi", source="bls", scheduled_at=ts),
        _ev(event_id="ecb:rate", source="ecb", scheduled_at=ts),
    ])
    got = store.get_events_between(ts - timedelta(days=1), ts + timedelta(days=1))
    assert {e.source for e in got} == {"bls", "ecb"}
    assert len(got) == 2
