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
        assert e.value_unit, f"{e.event_id} missing unit"
        assert e.periodicity in {"monthly", "quarterly", "eight_per_year"}
        assert e.license_label
        assert e.scheduled_at.tzinfo is not None
        assert e.time_confirmed is True


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
    # The three Eurostat euro events have no reliably-verified dates yet (calendar
    # served in JS), so they must NOT appear — an honest gap, never a fabricated
    # date. They are populated by the live .ics feed or a later verification.
    keys = {e.provider_ref.split(":")[0] for e in OfficialCalendarProvider().fetch().events}
    for absent in ("ea_hicp_flash", "ea_gdp_flash", "ea_unemployment"):
        assert absent not in keys
