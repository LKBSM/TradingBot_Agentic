"""NW-9 — the FOMC HTML calendar parser + date source.

The Federal Reserve publishes no .ics feed, so FOMC rate-decision DATES come from
the HTML calendar. These tests pin the row parser against a fixed fixture,
including the cross-month meeting (April 30 – May 1 → decision May 1) and the
year-panel split.
"""

from __future__ import annotations

from src.intelligence.calendar_providers.official_sources.base_official import (
    load_catalog,
)
from src.intelligence.calendar_providers.official_sources.fomc_schedule import (
    fomc_date_source,
    parse_fomc_calendar,
)


def _meeting(month: str, days: str) -> str:
    return (
        '<div class="row fomc-meeting">'
        f'<div class="fomc-meeting__month col-xs-5"><strong>{month}</strong></div>'
        f'<div class="fomc-meeting__date col-xs-4">{days}</div>'
        "</div>"
    )


FIXTURE = (
    '<div class="panel"><div class="panel-heading"><h4><a id="1">2025 FOMC Meetings</a></h4></div>'
    + _meeting("January", "28-29")
    + _meeting("March", "18-19")
    + _meeting("April/May", "29-30")   # NOT a real 2025 pairing — exercises cross-month
    + "</div>"
    + '<div class="panel"><div class="panel-heading"><h4><a id="2">2024 FOMC Meetings</a></h4></div>'
    + _meeting("April/May", "30-1")    # decision falls on May 1
    + _meeting("December", "17-18")
    + "</div>"
)


def test_parse_takes_second_day_and_second_month():
    dates = parse_fomc_calendar(FIXTURE)
    assert "2025-01-29" in dates          # second day of a same-month meeting
    assert "2025-03-19" in dates
    assert "2024-05-01" in dates          # cross-month: April 30 – May 1 → May 1
    assert "2024-12-18" in dates
    # The April/May 29-30 row resolves into May.
    assert "2025-05-30" in dates
    assert dates == sorted(set(dates))    # sorted + unique


def test_parse_is_graceful_on_junk():
    assert parse_fomc_calendar("") == []
    assert parse_fomc_calendar("<html>no meetings</html>") == []


def test_date_source_dates_only_the_rate_decision():
    src = fomc_date_source("federal_reserve", fetch_fn=lambda _url: FIXTURE)
    instances = src(load_catalog())
    assert instances, "expected FOMC decision instances"
    assert {i.event_key for i in instances} == {"us_fomc_rate"}
    assert "2024-05-01" in {i.release_date for i in instances}


def test_date_source_empty_when_feed_fails():
    src = fomc_date_source("federal_reserve", fetch_fn=lambda _url: "")
    assert src(load_catalog()) == []
