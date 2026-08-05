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
from src.intelligence.calendar_providers.values.bea_values import BEAValueFetcher
from src.intelligence.calendar_providers.values.census_values import CensusValueFetcher


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
# BEA NIPA
# --------------------------------------------------------------------------- #
_BEA_OK = json.dumps({
    "BEAAPI": {"Results": {"Data": [
        {"LineNumber": "1", "TimePeriod": "2026Q1", "DataValue": "2.4"},
        {"LineNumber": "1", "TimePeriod": "2026Q2", "DataValue": "3.1"},
        {"LineNumber": "2", "TimePeriod": "2026Q2", "DataValue": "9.9"},
    ]}}
})


def test_bea_fetch_reads_headline_line_actual_and_previous():
    f = BEAValueFetcher(api_key="KEY", http_get=lambda url: _BEA_OK)
    vp = f.fetch("NIPA-T10101")
    assert vp is not None
    # Chronological: Q2 latest (line 1 only), Q1 previous — line 2 ignored.
    assert vp.actual == 3.1
    assert vp.previous == 2.4


def test_bea_strips_thousands_comma_as_published():
    payload = json.dumps({"BEAAPI": {"Results": {"Data": [
        {"LineNumber": "1", "TimePeriod": "2026M05", "DataValue": "1,234.5"},
        {"LineNumber": "1", "TimePeriod": "2026M06", "DataValue": "1,240.0"},
    ]}}})
    f = BEAValueFetcher(api_key="KEY", http_get=lambda url: payload)
    vp = f.fetch("NIPA-T20804")
    assert vp is not None and vp.actual == 1240.0 and vp.previous == 1234.5


def test_bea_without_key_returns_none():
    f = BEAValueFetcher(api_key="", http_get=lambda url: _BEA_OK)
    assert f.fetch("NIPA-T10101") is None


def test_bea_unknown_series_returns_none():
    f = BEAValueFetcher(api_key="KEY", http_get=lambda url: _BEA_OK)
    assert f.fetch("NIPA-TZZZZZ") is None


def test_bea_graceful_on_error_and_unreachable():
    err = json.dumps({"BEAAPI": {"Results": {"Error": {"APIErrorCode": "1"}}}})
    assert BEAValueFetcher(api_key="KEY", http_get=lambda url: err).fetch("NIPA-T10101") is None
    # Unreachable → "" → None, and the cache is never touched (fetch just returns None).
    assert BEAValueFetcher(api_key="KEY", http_get=lambda url: "").fetch("NIPA-T10101") is None


# --------------------------------------------------------------------------- #
# Census EITS
# --------------------------------------------------------------------------- #
_CENSUS_OK = json.dumps([
    ["cell_value", "time", "us"],
    ["612345", "2026-05", "1"],
    ["615000", "2026-06", "1"],
])


def test_census_fetch_actual_and_previous_chronological():
    f = CensusValueFetcher(api_key="KEY", http_get=lambda url: _CENSUS_OK)
    vp = f.fetch("MARTS-RSAFS")
    assert vp is not None
    assert vp.actual == 615000.0
    assert vp.previous == 612345.0


def test_census_without_key_returns_none():
    f = CensusValueFetcher(api_key="", http_get=lambda url: _CENSUS_OK)
    assert f.fetch("MARTS-RSAFS") is None


def test_census_unknown_series_returns_none():
    f = CensusValueFetcher(api_key="KEY", http_get=lambda url: _CENSUS_OK)
    assert f.fetch("MARTS-ZZZZ") is None


def test_census_graceful_on_missing_key_html_page():
    # The keyless "Missing Key" response is HTML, not JSON → parsed to None,
    # never a fabricated value; the event stays unfetched.
    f = CensusValueFetcher(api_key="KEY", http_get=lambda url: "<html>Missing Key</html>")
    assert f.fetch("MARTS-RSAFS") is None


_CENSUS_INVALID_KEY = (
    '<html style="font-size: 14px;"><head><title>Invalid Key</title></head>'
    "<body>The key you provided is invalid.</body></html>"
)


def test_census_invalid_key_logs_actionable_warning(caplog):
    # A rejected/unactivated key answers HTML at HTTP 200 — must be logged
    # distinctly (not silently indistinguishable from "no data"), so a
    # misconfigured key surfaces in the operator's logs.
    f = CensusValueFetcher(api_key="BADKEY", http_get=lambda url: _CENSUS_INVALID_KEY)
    with caplog.at_level("WARNING"):
        assert f.fetch("MARTS-RSAFS") is None
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "CENSUS_API_KEY" in msg and "ACTIVATED" in msg


def test_census_is_key_error_detects_gate_pages():
    from src.intelligence.calendar_providers.values.census_values import _is_key_error

    assert _is_key_error("<html><title>Invalid Key</title></html>") is True
    assert _is_key_error("<!doctype html><body>Missing Key</body>") is True
    # A real JSON array (or any non-gate body) is NOT a key error.
    assert _is_key_error('[["cell_value","time"],["1","2026-06"]]') is False
    assert _is_key_error("") is False


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


def test_bea_and_census_wired_only_when_their_key_present(monkeypatch):
    monkeypatch.setenv("CALENDAR_VALUES_LIVE", "1")
    monkeypatch.delenv("BEA_API_KEY", raising=False)
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    f = build_value_fetcher()
    assert f is not None
    assert "bea" not in f._by_source                   # type: ignore[attr-defined]
    assert "census" not in f._by_source                # type: ignore[attr-defined]

    monkeypatch.setenv("BEA_API_KEY", "k1")
    monkeypatch.setenv("CENSUS_API_KEY", "k2")
    f2 = build_value_fetcher()
    assert f2 is not None
    assert "bea" in f2._by_source                      # type: ignore[attr-defined]
    assert "census" in f2._by_source                   # type: ignore[attr-defined]
