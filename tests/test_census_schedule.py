"""NW-9 — the Census HTML release-calendar parser + date source.

Census publishes no .ics feed, so its release DATES come from the HTML list-view
calendar. These tests pin the row parser and the catalog-keyword mapping against a
fixed HTML fixture (no network), including the two look-alike traps that must NOT
map: "New Residential Sales" (≠ housing starts) and the "Advance Economic
Indicators Report" that also contains the word "Retail" (≠ advance retail sales).
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.intelligence.calendar_providers.official_sources.base_official import (
    load_catalog,
)
from src.intelligence.calendar_providers.official_sources.census_schedule import (
    census_calendar_urls,
    census_date_source,
    parse_census_calendar,
)


def _row(name: str, key: str, human: str, time: str = "8:30 AM") -> str:
    return (
        '<tr height="20">'
        f'<td height="20"><a href="/x">{name}</a></td>'
        f'<td sorttable_customkey="{key}">{human}</td>'
        f"<td>{time}</td><td>reference</td>"
        f'<td class="hiden">A{key}</td>'
        "</tr>"
    )


FIXTURE = (
    "<table>"
    + _row(
        "Advance Report on Durable Goods--Manufacturers' Shipments, "
        "Inventories, and Orders",
        "202503260830",
        "March 26, 2025",
    )
    + _row(
        "Advance Monthly Sales for Retail and Food Services",
        "202503140830",
        "March 14, 2025",
    )
    + _row(
        "New Residential Construction (Building Permits, Housing Starts, "
        "and Housing Completions)",
        "202503180830",
        "March 18, 2025",
    )
    # Trap 1 — new home SALES, not housing starts.
    + _row("New Residential Sales", "202503241000", "March 24, 2025", "10:00 AM")
    # Trap 2 — a different report that merely contains the word "Retail".
    + _row(
        "Advance Economic Indicators Report (International Trade, Retail, "
        "& Wholesale Inventories)",
        "202503270830",
        "March 27, 2025",
    )
    # Unrelated indicator with no catalog mapping.
    + _row("Construction Spending (Construction Put in Place)", "202503031000", "March 3, 2025", "10:00 AM")
    # A row with no sortable key is ignored (never a fabricated date).
    + "<tr><td><a href='/x'>Durable Goods no-key row</a></td><td>March 26, 2025</td></tr>"
    + "</table>"
)


def test_parse_extracts_name_and_iso_date_from_customkey():
    pairs = parse_census_calendar(FIXTURE)
    names = {n for n, _ in pairs}
    assert ("Advance Report on Durable Goods--Manufacturers' Shipments, Inventories, and Orders", "2025-03-26") in pairs
    assert ("Advance Monthly Sales for Retail and Food Services", "2025-03-14") in pairs
    # The keyless row contributes nothing.
    assert not any(n == "Durable Goods no-key row" for n, _ in pairs)
    assert "New Residential Sales" in names  # parsed, but must not MAP (below)


def test_parse_is_graceful_on_junk():
    assert parse_census_calendar("") == []
    assert parse_census_calendar("<html>no rows here</html>") == []


def test_date_source_maps_only_the_three_measured_indicators():
    catalog = load_catalog()
    src = census_date_source("census", fetch_fn=lambda _url: FIXTURE, years_back=0)
    instances = src(catalog)
    by_key: dict[str, list[str]] = {}
    for inst in instances:
        by_key.setdefault(inst.event_key, []).append(inst.release_date)

    assert by_key.get("us_durable_goods") == ["2025-03-26"]
    assert by_key.get("us_retail_sales") == ["2025-03-14"]
    assert by_key.get("us_housing_starts") == ["2025-03-18"]
    # Traps did not leak into the measured keys.
    assert set(by_key) == {"us_durable_goods", "us_retail_sales", "us_housing_starts"}


def test_date_source_empty_when_feed_fails():
    src = census_date_source("census", fetch_fn=lambda _url: "", years_back=0)
    assert src(load_catalog()) == []


def test_calendar_urls_are_current_year_plus_archives():
    now = datetime(2026, 8, 7, tzinfo=timezone.utc)
    urls = census_calendar_urls(now=now, years_back=2)
    assert urls[0].endswith("calendar-listview.html")
    assert any(u.endswith("calendar-listview-2025.html") for u in urls)
    assert any(u.endswith("calendar-listview-2024.html") for u in urls)
    assert len(urls) == 3
