"""Calendar adapters (NW-1). The ForexFactory adapter is a dev-only prototype;
these tests pin its neutral mapping (all impacts kept, FF-only fields left None,
provenance/license stamped)."""

from __future__ import annotations

from src.intelligence.calendar_providers.forexfactory_provider import (
    ForexFactoryCalendarProvider,
)

_RAW = [
    {"title": "CPI y/y", "country": "USD", "impact": "high",
     "date": "2026-07-28T08:30:00-04:00", "actual": "", "forecast": "3.1%", "previous": "3.3%"},
    {"title": "Consumer Confidence", "country": "USD", "impact": "low",
     "date": "2026-07-28T10:00:00-04:00", "forecast": "99.1", "previous": "98.0"},
    {"title": "Bank Holiday", "country": "EUR", "impact": "holiday",
     "date": "2026-07-28T00:00:00-04:00"},
    {"title": "Malformed", "country": "", "impact": "high", "date": ""},
]


def _provider():
    return ForexFactoryCalendarProvider(fetch_fn=lambda: list(_RAW))


def test_keeps_all_impacts_including_low():
    events = _provider().fetch().events
    impacts = {e.impact for e in events}
    assert "low" in impacts and "high" in impacts


def test_drops_holiday_and_malformed():
    events = _provider().fetch().events
    titles = {e.event for e in events}
    assert "Bank Holiday" not in titles
    assert "Malformed" not in titles
    assert titles == {"CPI y/y", "Consumer Confidence"}


def test_ff_only_fields_are_none_never_fabricated():
    ev = next(e for e in _provider().fetch().events if e.event == "CPI y/y")
    assert ev.series_code is None       # no stable recurring id from FF
    assert ev.organism is None          # FF names no issuing organism
    assert ev.value_unit is None        # FF carries no clean unit


def test_provenance_and_license_stamped():
    ev = _provider().fetch().events[0]
    assert ev.source == "forexfactory"
    assert ev.source_timezone == "America/New_York"
    assert ev.license_label and "commercial" in ev.license_label.lower()
    assert ev.provider_ref  # deterministic native ref present


def test_values_parsed_numerically():
    ev = next(e for e in _provider().fetch().events if e.event == "CPI y/y")
    assert ev.forecast == 3.1
    assert ev.previous == 3.3
    assert ev.actual is None  # empty string → None, not 0


def test_coverage_window_from_events():
    fetch = _provider().fetch()
    assert fetch.coverage_start is not None
    assert fetch.coverage_end is not None
    assert fetch.coverage_start <= fetch.coverage_end
