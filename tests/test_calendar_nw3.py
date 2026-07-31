"""NW-3 backend tests — value SERIES (twelve-figure curve) + month range.

Covers the additions NW-3 makes to the calendar layer:
  · Eurostat serves a full ordered series of (period, value) — the curve source.
  · The value-fetch dispatcher routes a series request and degrades to [].
  · CalendarService.get_event attaches the series ONLY on the detail path.
  · CalendarService.get_calendar_range serves an arbitrary month window.

No network: the Eurostat fetcher takes an injected http_get. No real price data
is touched here (measures have their own pure test).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.intelligence.calendar_providers.values.base_value import (
    MultiValueFetcher,
    SeriesPoint,
    ValueFetcher,
    ValuePoint,
)
from src.intelligence.calendar_providers.values.eurostat_values import (
    EurostatValueFetcher,
    _parse_jsonstat_points,
)
from src.intelligence.calendar_service import CalendarService
from src.storage.calendar_cache_store import CalendarCacheEvent, CalendarCacheStore

_JSONSTAT = json.dumps(
    {
        "value": {"0": 2.1, "1": 2.4, "2": 2.0},
        "dimension": {
            "time": {"category": {"index": {"2026-04": 0, "2026-05": 1, "2026-06": 2}}}
        },
    }
)


def test_parse_jsonstat_points_orders_by_time_index():
    pts = _parse_jsonstat_points(_JSONSTAT)
    assert [(p.period, p.value) for p in pts] == [
        ("2026-04", 2.1),
        ("2026-05", 2.4),
        ("2026-06", 2.0),
    ]


def test_parse_jsonstat_points_empty_on_garbage():
    assert _parse_jsonstat_points("not json") == []
    assert _parse_jsonstat_points(json.dumps({"value": {}})) == []


def test_eurostat_fetch_series_returns_ordered_points():
    fetcher = EurostatValueFetcher(http_get=lambda url: _JSONSTAT)
    pts = fetcher.fetch_series("prc_hicp_manr", limit=12)
    assert len(pts) == 3
    assert pts[-1] == SeriesPoint(period="2026-06", value=2.0)


def test_eurostat_fetch_series_unknown_dataset_is_empty():
    fetcher = EurostatValueFetcher(http_get=lambda url: _JSONSTAT)
    assert fetcher.fetch_series("unknown_code") == []


class _StubSeriesFetcher(ValueFetcher):
    def fetch(self, series_code):  # not used here
        return ValuePoint(actual=2.0, previous=2.1)

    def fetch_series(self, series_code, limit=12):
        return [SeriesPoint(period="2026-05", value=2.1), SeriesPoint(period="2026-06", value=2.0)]


class _NoSeriesFetcher(ValueFetcher):
    def fetch(self, series_code):
        return None
    # inherits the default fetch_series -> []


def test_multi_value_fetcher_series_routes_and_degrades():
    mv = MultiValueFetcher({"eurostat": _StubSeriesFetcher(), "ecb": _NoSeriesFetcher()})
    assert len(mv.series_for("eurostat", "prc_hicp_manr")) == 2
    assert mv.series_for("ecb", "any") == []          # source serves no series
    assert mv.series_for("bls", "x") == []            # no registered fetcher
    assert mv.series_for("eurostat", None) == []      # no series code


class _MinimalProvider:
    source_name = "eurostat"

    def attributions(self):
        return []

    def fetch(self):  # never reached: within TTL after the seed upsert
        raise AssertionError("provider.fetch should not run within TTL")


def _seed_store(tmp_path) -> CalendarCacheStore:
    store = CalendarCacheStore(db_path=str(tmp_path / "cal.db"))
    now = datetime(2026, 6, 15, 9, 0, tzinfo=timezone.utc)
    store.upsert_events(
        [
            CalendarCacheEvent(
                event_id="eurostat:ea_hicp_flash:2026-06-30",
                source="eurostat",
                event="Inflation IPCH zone euro",
                currency="EUR",
                scheduled_at=datetime(2026, 6, 30, 9, 0, tzinfo=timezone.utc),
                markets=["EURUSD"],
                series_code="prc_hicp_manr",
                organism="Eurostat",
                periodicity="monthly",
            )
        ],
        fetched_at=now,
    )
    return store


def test_get_event_attaches_value_series_on_detail_path(tmp_path):
    store = _seed_store(tmp_path)
    service = CalendarService(
        provider=_MinimalProvider(),
        store=store,
        market_map={"EURUSD": ["EUR", "USD"]},
        value_fetcher=MultiValueFetcher({"eurostat": _StubSeriesFetcher()}),
    )
    now = datetime(2026, 6, 15, 9, 0, 30, tzinfo=timezone.utc)
    resp = service.get_event("eurostat:ea_hicp_flash:2026-06-30", now=now)
    assert len(resp.events) == 1
    series = resp.events[0].value_series
    assert [(p.period, p.value) for p in series] == [("2026-05", 2.1), ("2026-06", 2.0)]


def test_get_calendar_range_serves_a_month_window(tmp_path):
    store = _seed_store(tmp_path)
    service = CalendarService(
        provider=_MinimalProvider(),
        store=store,
        market_map={"EURUSD": ["EUR", "USD"]},
        value_fetcher=None,  # range/list path never attaches a series
    )
    now = datetime(2026, 6, 15, 9, 0, 30, tzinfo=timezone.utc)
    start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    end = datetime(2026, 6, 30, 23, 59, 59, tzinfo=timezone.utc)
    resp = service.get_calendar_range(start, end, now=now)
    assert [e.event_id for e in resp.events] == ["eurostat:ea_hicp_flash:2026-06-30"]
    # the list/range path must NOT attach a per-event series (one call per detail)
    assert resp.events[0].value_series == []

    # a month with no events yields an empty, honest window (not an error)
    may = service.get_calendar_range(
        datetime(2026, 5, 1, tzinfo=timezone.utc),
        datetime(2026, 5, 31, 23, 59, 59, tzinfo=timezone.utc),
        now=now,
    )
    assert may.events == []
