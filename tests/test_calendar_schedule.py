"""Guard for the versioned, auditable release schedule (NW-1b).

Asserts the shipped ``config/calendar_schedule.json`` is coherent with the
catalog and that the DEFAULT official provider (reading the real config, no
injection) serves real, attributed events — i.e. the calendar is client-facing,
not empty. Every schedule row must be auditable (schedule_url + last_verified)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from src.intelligence.calendar_providers.official_provider import OfficialCalendarProvider
from src.intelligence.calendar_providers.official_sources.base_official import (
    load_catalog,
    load_schedule,
)

_CONFIG = Path(__file__).resolve().parents[1] / "config"


def _raw_schedule() -> dict:
    return json.loads((_CONFIG / "calendar_schedule.json").read_text(encoding="utf-8"))


def test_every_schedule_key_exists_in_catalog() -> None:
    catalog = load_catalog()
    for inst in load_schedule():
        assert inst.event_key in catalog, f"unknown schedule key: {inst.event_key}"


def test_every_release_row_is_auditable() -> None:
    # Each dated release must cite its official schedule URL + a verification date.
    for r in _raw_schedule()["releases"]:
        assert r.get("event_key")
        assert r.get("date") and date.fromisoformat(r["date"])  # valid ISO date
        assert str(r.get("schedule_url", "")).startswith("https://")
        assert r.get("last_verified") and date.fromisoformat(r["last_verified"])


def test_default_provider_serves_real_events_not_empty() -> None:
    # No injection: the shipped catalog + schedule must yield real events, so the
    # client-facing calendar is populated rather than honestly-empty.
    fetch = OfficialCalendarProvider().fetch()
    assert len(fetch.events) >= 20
    sources = {e.source for e in fetch.events}
    # the biggest drivers are wired: US macro + ECB rate
    assert {"bls", "bea", "census", "federal_reserve", "ecb"} <= sources


def test_served_events_carry_full_official_shape() -> None:
    for e in OfficialCalendarProvider().fetch().events:
        assert e.organism, f"{e.event_id} missing organism"
        # A MEASURABLE event (one with a data series) must declare its unit; a
        # moment-only event (no series — FOMC minutes, dot plot) has no numeric
        # value and therefore no unit, honestly (NW-4 ch.4). Never a fake unit.
        if e.series_code:
            assert e.value_unit, f"{e.event_id} missing unit"
        assert e.periodicity in {"monthly", "quarterly", "eight_per_year"}
        assert e.license_label
        assert e.scheduled_at.tzinfo is not None
        # The publication TIME may be confirmed or not — both are honest (NW-4):
        # an unconfirmed time is flagged, never approximated or hidden. The field
        # is always a real boolean; the "confirmed ⇒ proof" invariant is enforced
        # at load and covered by test_calendar_time_confirmation_guard.
        assert isinstance(e.time_confirmed, bool)


def test_scheduled_times_are_dst_correct() -> None:
    events = {e.provider_ref: e for e in OfficialCalendarProvider().fetch().events}
    # BLS Employment Situation 2026-08-07 at 08:30 America/New_York (EDT) → 12:30Z
    nfp = events.get("us_employment_situation:2026-08-07")
    assert nfp is not None
    assert nfp.scheduled_at == datetime(2026, 8, 7, 12, 30, tzinfo=timezone.utc)
    # ECB decision 2026-09-10 at 14:15 Europe/Berlin (CEST) → 12:15Z
    ecb = events.get("ea_ecb_rate:2026-09-10")
    assert ecb is not None
    assert ecb.scheduled_at == datetime(2026, 9, 10, 12, 15, tzinfo=timezone.utc)


def test_uncovered_events_are_absent_not_fabricated() -> None:
    # Only euro-area unemployment lacks a reliably-verified date (Eurostat calendar
    # served in JS), so it must NOT appear — an honest gap, never a fabricated date.
    # HICP flash and GDP flash are now populated from official Eurostat pages.
    keys = {e.provider_ref.split(":")[0] for e in OfficialCalendarProvider().fetch().events}
    assert "ea_unemployment" not in keys
    assert "ea_hicp_flash" in keys      # populated from the euro-indicators calendar
    assert "ea_gdp_flash" in keys       # populated from the QNA release calendar


def test_nw4_added_events_present_and_honestly_shaped() -> None:
    cat = load_catalog()
    # JOLTS — BLS series, released at 10:00 ET (not 08:30), measurable (has unit).
    assert cat["us_jolts"].series_code
    assert cat["us_jolts"].release_time_local == "10:00"
    assert cat["us_jolts"].value_unit
    # Core CPI — its OWN BLS series, distinct from the headline.
    assert cat["us_cpi_core"].series_code == "CUUR0000SA0L1E"
    # Moment-only Fed events — no series, no unit (honest), time confirmed w/ proof.
    for k in ("us_fomc_minutes", "us_fomc_dotplot"):
        assert cat[k].source == "federal_reserve"
        assert cat[k].series_code is None
        assert cat[k].value_unit is None
        assert cat[k].time_confirmed is True


def test_undatable_events_are_signalled_never_silent() -> None:
    # NW-4 ch.5A: an event that no wired source can date must be SURFACED, not
    # allowed to vanish silently. undatable_events() reports it (and fetch() logs
    # it). A datable event is never in the list.
    provider = OfficialCalendarProvider()
    undatable = provider.undatable_events()
    assert "ea_unemployment" in undatable      # honest gap, now explicitly reported
    assert "us_cpi" not in undatable           # datable via the curated schedule
    # Every reported key is a real catalog event (never a phantom).
    catalog = load_catalog()
    assert all(k in catalog for k in undatable)
