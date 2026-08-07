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


# --- BLS fetch_series (NW-6): monthly curve, chronological, month-only ---------
_BLS_SERIES_OK = json.dumps({
    "status": "REQUEST_SUCCEEDED",
    "Results": {"series": [{"data": [
        # Most-recent-first, as BLS returns it; plus an annual-average row (M13).
        {"year": "2026", "period": "M13", "value": "999.9"},
        {"year": "2026", "period": "M03", "value": "322.9"},
        {"year": "2026", "period": "M02", "value": "321.5"},
        {"year": "2026", "period": "M01", "value": "320.0"},
        {"year": "2025", "period": "M12", "value": "319.1"},
    ]}]},
})


def test_bls_fetch_series_is_chronological_and_labelled():
    f = BLSValueFetcher(api_key="KEY", http_post=lambda url, body: _BLS_SERIES_OK)
    pts = f.fetch_series("CUUR0000SA0")
    # Oldest→newest, month labels "YYYY-MM", M13 annual-average dropped.
    assert [(p.period, p.value) for p in pts] == [
        ("2025-12", 319.1),
        ("2026-01", 320.0),
        ("2026-02", 321.5),
        ("2026-03", 322.9),
    ]


def test_bls_fetch_series_respects_limit_keeping_most_recent():
    f = BLSValueFetcher(api_key="KEY", http_post=lambda url, body: _BLS_SERIES_OK)
    pts = f.fetch_series("CUUR0000SA0", limit=2)
    assert [(p.period, p.value) for p in pts] == [("2026-02", 321.5), ("2026-03", 322.9)]


# --- BLS fetch_series index_change / count_change (NW-8): published variations ---
_BLS_SERIES_CALC = json.dumps({
    "status": "REQUEST_SUCCEEDED",
    "Results": {"series": [{"data": [
        {"year": "2026", "period": "M03", "value": "322.9",
         "calculations": {"pct_changes": {"1": "0.4", "12": "3.1"}}},
        {"year": "2026", "period": "M02", "value": "321.5",
         "calculations": {"pct_changes": {"1": "0.5", "12": "3.4"}}},
        # A month WITHOUT a 12-month calculation (e.g. the 2025 lapse gap) → no point.
        {"year": "2026", "period": "M01", "value": "320.0",
         "calculations": {"pct_changes": {"1": "0.3"}}},
    ]}]},
})

_BLS_SERIES_COUNT = json.dumps({
    "status": "REQUEST_SUCCEEDED",
    "Results": {"series": [{"data": [
        {"year": "2026", "period": "M03", "value": "159000",
         "calculations": {"net_changes": {"1": "150"}}},
        {"year": "2026", "period": "M02", "value": "158850",
         "calculations": {"net_changes": {"1": "-30"}}},
    ]}]},
})


def test_bls_fetch_series_index_change_carries_yoy_level_and_mom():
    sent = {}

    def _post(url, body):
        sent["body"] = body
        return _BLS_SERIES_CALC

    f = BLSValueFetcher(api_key="KEY", http_post=_post)
    pts = f.fetch_series("CUUR0000SA0", kind="index_change")
    # value = published 12-month %; level = raw index; change_mom = published 1-month %.
    assert [(p.period, p.value, p.level, p.change_mom) for p in pts] == [
        ("2026-02", 3.4, 321.5, 0.5),
        ("2026-03", 3.1, 322.9, 0.4),
    ]
    # The request asked BLS to compute the changes (never recomputed locally).
    assert '"calculations": true' in sent["body"]


def test_derive_variation_amount_is_monthly_percent_from_levels():
    from src.intelligence.calendar_providers.values.base_value import (
        SeriesPoint, derive_variation_series,
    )
    levels = [SeriesPoint(period=f"2025-{m:02d}", value=100.0 + m) for m in range(1, 4)]
    out = derive_variation_series(levels, "amount")
    # First point dropped (no prior month); value = (level_t/level_{t-1} - 1)*100.
    assert [p.period for p in out] == ["2025-02", "2025-03"]
    assert round(out[-1].value, 4) == round((103.0 / 102.0 - 1) * 100, 4)
    assert out[-1].level == 103.0 and out[-1].change_mom is None


def test_derive_variation_index_carries_yoy_level_and_mom():
    from src.intelligence.calendar_providers.values.base_value import (
        SeriesPoint, derive_variation_series,
    )
    levels = [SeriesPoint(period=str(i), value=100.0 + i) for i in range(14)]
    out = derive_variation_series(levels, "index")
    # Points before the 12th are dropped (no year-ago level).
    assert [p.period for p in out] == ["12", "13"]
    last = out[-1]  # level 113 vs 101 a year earlier, 112 a month earlier
    assert round(last.value, 4) == round((113.0 / 101.0 - 1) * 100, 4)  # yoy
    assert round(last.change_mom, 4) == round((113.0 / 112.0 - 1) * 100, 4)  # mo
    assert last.level == 113.0


def test_bls_fetch_series_count_change_is_absolute_monthly_change():
    f = BLSValueFetcher(api_key="KEY", http_post=lambda url, body: _BLS_SERIES_COUNT)
    pts = f.fetch_series("CES0000000001", kind="count_change")
    # value = published 1-month absolute change; level = the raw total count.
    assert [(p.period, p.value, p.level) for p in pts] == [
        ("2026-02", -30.0, 158850.0),
        ("2026-03", 150.0, 159000.0),
    ]


def test_bls_fetch_series_level_ignores_calculations():
    f = BLSValueFetcher(api_key="KEY", http_post=lambda url, body: _BLS_SERIES_CALC)
    pts = f.fetch_series("CUUR0000SA0", kind="level")
    # Level mode keeps the index values, including the month without a % change.
    assert [(p.period, p.value) for p in pts] == [
        ("2026-01", 320.0), ("2026-02", 321.5), ("2026-03", 322.9),
    ]


def test_bls_fetch_series_without_key_returns_empty():
    f = BLSValueFetcher(api_key="", http_post=lambda url, body: _BLS_SERIES_OK)
    assert f.fetch_series("CUUR0000SA0") == []


def test_bls_fetch_series_graceful_on_error_payload():
    bad = json.dumps({"status": "REQUEST_NOT_PROCESSED", "Results": {}})
    f = BLSValueFetcher(api_key="KEY", http_post=lambda url, body: bad)
    assert f.fetch_series("CUUR0000SA0") == []


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


# --- ECB fetch_series (NW-6): distinct rate-level change points ----------------
def _ecb_series_payload(pairs):
    """(date, value) pairs → an SDMX-JSON message with a dated TIME dimension."""
    obs = {str(i): [v] for i, (_, v) in enumerate(pairs)}
    time_vals = [{"id": d} for d, _ in pairs]
    return json.dumps({
        "dataSets": [{"series": {"0:0": {"observations": obs}}}],
        "structure": {"dimensions": {"observation": [
            {"id": "TIME_PERIOD", "values": time_vals}
        ]}},
    })


def test_ecb_fetch_series_collapses_runs_to_change_dates():
    # Held 2.15, stepped to 2.40, held, stepped to 2.65: three decisions.
    payload = _ecb_series_payload([
        ("2026-04-10", 2.15), ("2026-05-10", 2.15),
        ("2026-06-06", 2.40), ("2026-07-06", 2.40),
        ("2026-08-05", 2.65),
    ])
    f = ECBValueFetcher(http_get=lambda url: payload)
    pts = f.fetch_series("FM.D.U2.EUR.4F.KR.MRR_FR.LEV")
    # One point per DISTINCT level, labelled by the month it took effect (YYYY-MM).
    assert [(p.period, p.value) for p in pts] == [
        ("2026-04", 2.15), ("2026-06", 2.40), ("2026-08", 2.65),
    ]


def test_ecb_fetch_series_empty_without_structure():
    # A message lacking the TIME dimension yields no curve (never fabricated).
    payload = json.dumps({"dataSets": [{"series": {"0:0": {"observations": {"0": [2.4]}}}}]})
    f = ECBValueFetcher(http_get=lambda url: payload)
    assert f.fetch_series("FM.D.U2.EUR.4F.KR.MRR_FR.LEV") == []


def test_ecb_fetch_series_empty_on_failure():
    f = ECBValueFetcher(http_get=lambda url: "")
    assert f.fetch_series("FM.D.U2.EUR.4F.KR.MRR_FR.LEV") == []


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
