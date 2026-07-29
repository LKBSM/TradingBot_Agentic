"""Live .ics feed parse + match + opt-in wiring (NW-1b).

Deterministic: no network — the parser takes text and the date source takes an
injected fetch. Verifies the SUMMARY→key mapping (incl. exclusions) and the
opt-in fallback to the curated schedule."""

from __future__ import annotations

from src.intelligence.calendar_providers.official_sources import base_official
from src.intelligence.calendar_providers.official_sources.base_official import (
    ics_date_source,
    load_catalog,
    match_ics_key,
)
from src.intelligence.calendar_providers.official_sources.ics_feed import parse_ics

# Real BEA-style feed: folded SUMMARY, escaped commas, DATE-TIME (Z) + DATE forms,
# plus a "by State" GDP line that must be excluded.
BEA_ICS = (
    "BEGIN:VCALENDAR\r\n"
    "BEGIN:VEVENT\r\n"
    "SUMMARY:Gross Domestic Product\\, 3rd Quarter 2026 (Advance Estima\r\n"
    " te)\r\n"
    "DTSTART;VALUE=DATE-TIME:20261029T123000Z\r\n"
    "UID:a\r\n"
    "END:VEVENT\r\n"
    "BEGIN:VEVENT\r\n"
    "SUMMARY:Gross Domestic Product by State\\, 2nd Quarter 2026\r\n"
    "DTSTART;VALUE=DATE-TIME:20261218T123000Z\r\n"
    "UID:b\r\n"
    "END:VEVENT\r\n"
    "BEGIN:VEVENT\r\n"
    "SUMMARY:Personal Income and Outlays\\, September 2026\r\n"
    "DTSTART;VALUE=DATE:20261029\r\n"
    "UID:c\r\n"
    "END:VEVENT\r\n"
    "END:VCALENDAR\r\n"
)


def test_parse_ics_unfolds_and_reads_summary_and_date():
    events = parse_ics(BEA_ICS)
    assert ("Gross Domestic Product, 3rd Quarter 2026 (Advance Estimate)", "2026-10-29") in events
    # date-only DTSTART is handled too
    assert ("Personal Income and Outlays, September 2026", "2026-10-29") in events
    assert len(events) == 3


def test_match_ics_key_maps_and_excludes():
    bea = {k: c for k, c in load_catalog().items() if c.source == "bea"}
    assert match_ics_key("Gross Domestic Product, 3rd Quarter 2026 (Advance Estimate)", bea) == "us_gdp"
    # "by State" is excluded → no match (never a false GDP instance)
    assert match_ics_key("Gross Domestic Product by State, 2nd Quarter 2026", bea) is None
    assert match_ics_key("Personal Income and Outlays, September 2026", bea) == "us_pce"


def test_cpi_ppi_disambiguated():
    bls = {k: c for k, c in load_catalog().items() if c.source == "bls"}
    assert match_ics_key("Consumer Price Index - August 2026", bls) == "us_cpi"
    assert match_ics_key("Producer Price Index - August 2026", bls) == "us_ppi"
    assert match_ics_key("Employment Situation - August 2026", bls) == "us_employment_situation"


def test_ics_date_source_yields_matched_releases():
    catalog = load_catalog()
    src = ics_date_source("bea", fetch_fn=lambda url: BEA_ICS)
    out = {(r.event_key, r.release_date) for r in src(catalog)}
    assert ("us_gdp", "2026-10-29") in out
    assert ("us_pce", "2026-10-29") in out
    # the "by State" line produced no release
    assert all(r.event_key == "us_gdp" or r.event_key == "us_pce" for r in src(catalog))


def test_ics_date_source_empty_on_fetch_failure():
    src = ics_date_source("bea", fetch_fn=lambda url: "")
    assert src(load_catalog()) == []


def test_default_uses_curated_when_flag_off(monkeypatch):
    monkeypatch.delenv("CALENDAR_ICS_LIVE", raising=False)
    from src.intelligence.calendar_providers.official_provider import (
        OfficialCalendarProvider,
    )
    # curated schedule serves (deterministic) — no network touched
    events = OfficialCalendarProvider().fetch().events
    assert len(events) >= 20


def test_default_prefers_live_ics_when_flag_on(monkeypatch):
    monkeypatch.setenv("CALENDAR_ICS_LIVE", "1")
    # Patch the network fetch so the live path is exercised without a socket.
    import src.intelligence.calendar_providers.official_sources.ics_feed as ics
    monkeypatch.setattr(ics, "fetch_ics", lambda url, timeout=10: BEA_ICS if "bea" in url else "")
    from src.intelligence.calendar_providers.official_sources.organisms import BEAProvider
    events = BEAProvider().fetch().events
    refs = {e.provider_ref for e in events}
    # live BEA dates win: GDP + PCE on 2026-10-29 from the feed
    assert "us_gdp:2026-10-29" in refs
    assert "us_pce:2026-10-29" in refs


def test_default_falls_back_to_curated_when_live_empty(monkeypatch):
    monkeypatch.setenv("CALENDAR_ICS_LIVE", "1")
    import src.intelligence.calendar_providers.official_sources.ics_feed as ics
    monkeypatch.setattr(ics, "fetch_ics", lambda url, timeout=10: "")  # feed down
    from src.intelligence.calendar_providers.official_sources.organisms import BEAProvider
    events = BEAProvider().fetch().events
    # curated BEA dates still served (source never erased by a feed failure)
    assert len(events) >= 3
    assert any(e.provider_ref.startswith("us_gdp:") for e in events)
