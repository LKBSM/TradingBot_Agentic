"""Per-organism value fetchers: parse + wiring (NW-1c §4).

Deterministic: HTTP is injected (no network). Covers the Eurostat JSON-stat
parser, the BLS v2 parser + key-gating, and the ECB step-series "previous =
last distinct" rule."""

from __future__ import annotations

import json

from src.intelligence.calendar_providers.values.base_value import build_value_fetcher
from src.intelligence.calendar_providers.values.bls_values import BLSValueFetcher
from src.intelligence.calendar_providers.values.ecb_values import (
    ECBValueFetcher,
    _last_distinct_before,
)
from src.intelligence.calendar_providers.values.eurostat_values import (
    EurostatValueFetcher,
    _parse_jsonstat,
)


# --------------------------------------------------------------------------- #
# Eurostat JSON-stat
# --------------------------------------------------------------------------- #
def test_eurostat_parse_jsonstat_orders_by_time():
    payload = json.dumps({
        "value": {"0": 2.1, "1": 2.0},
        "dimension": {"time": {"category": {"index": {"2026-06": 0, "2026-07": 1}}}},
    })
    assert _parse_jsonstat(payload) == [2.1, 2.0]


def test_eurostat_fetch_returns_actual_and_previous():
    payload = json.dumps({
        "value": {"0": 2.1, "1": 2.0},
        "dimension": {"time": {"category": {"index": {"2026-06": 0, "2026-07": 1}}}},
    })
    f = EurostatValueFetcher(http_get=lambda url: payload)
    vp = f.fetch("prc_hicp_manr")
    assert vp is not None
    assert vp.actual == 2.0
    assert vp.previous == 2.1


def test_eurostat_unknown_dataset_returns_none():
    f = EurostatValueFetcher(http_get=lambda url: "{}")
    assert f.fetch("not_a_dataset") is None


def test_eurostat_empty_on_failure():
    f = EurostatValueFetcher(http_get=lambda url: "")
    assert f.fetch("prc_hicp_manr") is None


# --------------------------------------------------------------------------- #
# BLS v2
# --------------------------------------------------------------------------- #
_BLS_OK = json.dumps({
    "status": "REQUEST_SUCCEEDED",
    "Results": {"series": [{"data": [
        {"year": "2026", "period": "M06", "value": "322.9"},
        {"year": "2026", "period": "M05", "value": "321.5"},
    ]}]},
})


def test_bls_parses_latest_two_most_recent_first():
    f = BLSValueFetcher(api_key="KEY", http_post=lambda url, body: _BLS_OK)
    vp = f.fetch("CUUR0000SA0")
    assert vp is not None
    assert vp.actual == 322.9
    assert vp.previous == 321.5


def test_bls_without_key_returns_none():
    # Key-gated: absent BLS_API_KEY it never calls out (stays unfetched).
    f = BLSValueFetcher(api_key="", http_post=lambda url, body: _BLS_OK)
    assert f.fetch("CUUR0000SA0") is None


def test_bls_graceful_on_error_payload():
    bad = json.dumps({"status": "REQUEST_NOT_PROCESSED", "Results": {}})
    f = BLSValueFetcher(api_key="KEY", http_post=lambda url, body: bad)
    assert f.fetch("CUUR0000SA0") is None


# --------------------------------------------------------------------------- #
# ECB step-series previous
# --------------------------------------------------------------------------- #
def test_last_distinct_before_on_step_series():
    # A rate that stepped 2.15 → 2.40 and held: previous is 2.15, not 2.40.
    obs = [2.15, 2.15, 2.15, 2.40, 2.40, 2.40]
    assert _last_distinct_before(obs, 2.40) == 2.15


def test_last_distinct_before_never_changed():
    assert _last_distinct_before([2.4, 2.4, 2.4], 2.4) is None


def test_ecb_previous_is_last_distinct():
    # Build an SDMX-JSON message where the rate held at 2.4 after stepping from 2.15.
    obs = {str(i): [v] for i, v in enumerate([2.15, 2.15, 2.40, 2.40])}
    payload = json.dumps({"dataSets": [{"series": {"0:0": {"observations": obs}}}]})
    f = ECBValueFetcher(http_get=lambda url: payload)
    vp = f.fetch("FM.D.U2.EUR.4F.KR.MRR_FR.LEV")
    assert vp is not None
    assert vp.actual == 2.40
    assert vp.previous == 2.15   # the distinct prior level, not the equal-neighbour


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #
def test_no_key_sources_wired_when_live(monkeypatch):
    monkeypatch.setenv("CALENDAR_VALUES_LIVE", "1")
    monkeypatch.delenv("BLS_API_KEY", raising=False)
    f = build_value_fetcher()
    assert f is not None
    # ECB + Eurostat are wired (no key); BLS is not (key-gated, absent key).
    assert f.fetch_for("eurostat", None) is None       # None series → None, but source is registered
    assert "eurostat" in f._by_source                  # type: ignore[attr-defined]
    assert "ecb" in f._by_source                       # type: ignore[attr-defined]
    assert "bls" not in f._by_source                   # type: ignore[attr-defined]


def test_bls_wired_when_key_present(monkeypatch):
    monkeypatch.setenv("CALENDAR_VALUES_LIVE", "1")
    monkeypatch.setenv("BLS_API_KEY", "test-key")
    f = build_value_fetcher()
    assert f is not None
    assert "bls" in f._by_source                       # type: ignore[attr-defined]
