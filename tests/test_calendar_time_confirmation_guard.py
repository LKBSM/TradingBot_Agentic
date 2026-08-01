"""NW-4 ch.1 — a confidence flag without proof is impossible.

``time_confirmed=true`` claims the publication TIME was verified on the
organism's official page. That claim is only trustworthy with evidence: the
page that documents the time (``time_source_url``) AND the date it was last
checked there (``time_last_verified``). These tests lock in:
  · the SHIPPED catalog never flags a time as confirmed without both fields;
  · ``load_catalog`` downgrades a confirmed flag that lacks either field;
  · the NW-4 corrections themselves (HICP flash → 15:00 confirmed; euro-area
    GDP flash & unemployment → honestly unconfirmed).
"""

import json
from pathlib import Path

from src.intelligence.calendar_providers.official_sources.base_official import load_catalog

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "config" / "calendar_catalog.json"


def _has_proof(ev: dict) -> bool:
    return bool(str(ev.get("time_source_url") or "").strip()) and bool(
        str(ev.get("time_last_verified") or "").strip()
    )


def test_shipped_catalog_confirmed_times_all_carry_proof():
    raw = json.loads(CATALOG.read_text(encoding="utf-8"))
    offenders = [
        ev.get("key")
        for ev in raw.get("events", [])
        if bool(ev.get("time_confirmed", False)) and not _has_proof(ev)
    ]
    assert offenders == [], f"time_confirmed=true without source_url+date: {offenders}"


def test_load_catalog_downgrades_confirmed_without_proof(tmp_path):
    cat = {
        "version": 1,
        "sources": {"bls": {"organism": "BLS", "license_label": "Public domain"}},
        "events": [
            {
                "key": "no_url", "source": "bls", "event": "X", "currency": "USD",
                "source_timezone": "America/New_York", "release_time_local": "08:30",
                "time_confirmed": True, "time_last_verified": "2026-08-01",
            },
            {
                "key": "no_date", "source": "bls", "event": "Y", "currency": "USD",
                "source_timezone": "America/New_York", "release_time_local": "08:30",
                "time_confirmed": True, "time_source_url": "https://example.gov/sched",
            },
            {
                "key": "empty_url", "source": "bls", "event": "W", "currency": "USD",
                "source_timezone": "America/New_York", "release_time_local": "08:30",
                "time_confirmed": True, "time_source_url": "   ",
                "time_last_verified": "2026-08-01",
            },
            {
                "key": "ok", "source": "bls", "event": "Z", "currency": "USD",
                "source_timezone": "America/New_York", "release_time_local": "08:30",
                "time_confirmed": True, "time_source_url": "https://example.gov/sched",
                "time_last_verified": "2026-08-01",
            },
        ],
    }
    p = tmp_path / "cat.json"
    p.write_text(json.dumps(cat), encoding="utf-8")
    loaded = load_catalog(p)

    # A confirmed flag without BOTH proof fields is downgraded to unconfirmed.
    assert loaded["no_url"].time_confirmed is False
    assert loaded["no_date"].time_confirmed is False
    assert loaded["empty_url"].time_confirmed is False
    # With both fields present, the flag stands.
    assert loaded["ok"].time_confirmed is True
    assert loaded["ok"].time_source_url == "https://example.gov/sched"
    assert loaded["ok"].time_last_verified == "2026-08-01"


def test_nw4_corrections_hicp_confirmed_gdp_and_unemployment_unconfirmed():
    cat = load_catalog(CATALOG)
    # HICP flash: corrected to the official 15:00 CET, confirmed with a real source.
    assert cat["ea_hicp_flash"].release_time_local == "15:00"
    assert cat["ea_hicp_flash"].time_confirmed is True
    # Euro-area GDP flash & unemployment: 11:00 is the supposed standard, not
    # anchored to a source page → honestly unconfirmed.
    assert cat["ea_gdp_flash"].time_confirmed is False
    assert cat["ea_unemployment"].time_confirmed is False
