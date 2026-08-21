"""Tests for POST /api/conditions-scan (read-only structural scan).

Covers: correct met/unmet over combos, partial (transparency) results, the
read-only guarantee (only get_latest_reading is touched — never a write nor
detection), predictive types rejected at the schema boundary, and 503 wiring.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import AppState
from src.api.routes.conditions_scan import (
    SCAN_COMBOS,
    _compute_freshness,
    router as scan_router,
)
from src.api.signal_store import SignalStore


def _reading(instrument, timeframe, *, close_price=2000.0, mtf=None, order_blocks=None, current_bos=None):
    return {
        "header": {
            "instrument": instrument,
            "timeframe": timeframe,
            "candle_close_ts": "2026-05-28T14:15:00+00:00",
            "close_price": close_price,
        },
        "structure": {
            "current_bos": current_bos,
            "current_choch": None,
            "bos_events": [],
            "choch_events": [],
            "order_blocks": order_blocks or [],
            "fair_value_gaps": [],
        },
        "regime": {
            "trend": "bullish",
            "volatility_observed": "normal",
            "market_phase": "trend",
            "mtf_confluence": mtf or {},
        },
        "events": {"news_upcoming": [], "news_just_published": [], "technical_triggers_recent": []},
        "conditions": {"tags": [], "description": "", "description_source": "engine_template"},
    }


def _ob(low, high):
    return {
        "id": "ob1",
        "direction": "bullish",
        "level_low": low,
        "level_high": high,
        "importance": "medium",
        "status": "active",
        "created_at": "2026-05-28T12:00:00+00:00",
        "tested": False,
        "user_flagged": False,
    }


class _RecordingStore:
    """Read-only store double. Writes raise, so any mutation fails the test."""

    def __init__(self, readings):
        self._readings = readings  # {(instrument, timeframe): payload | None}
        self.read_calls = []

    def get_latest_reading(self, instrument, timeframe):
        self.read_calls.append((instrument, timeframe))
        return self._readings.get((instrument, timeframe))

    def save_reading(self, *a, **k):  # pragma: no cover - must never be called
        raise AssertionError("scan must not write (save_reading called)")

    def mark_combination_active(self, *a, **k):  # pragma: no cover
        raise AssertionError("scan must not write (mark_combination_active called)")


class _RecordingAssembler:
    def __init__(self, store):
        self._store = store

    @property
    def readings_store(self):
        return self._store

    def get_or_generate(self, *a, **k):  # pragma: no cover - must never be called
        raise AssertionError("scan must not trigger detection (get_or_generate called)")


def _make_app(assembler=None, *, with_assembler=True):
    app = FastAPI()
    signal_store = SignalStore(
        db_path=str(tempfile.NamedTemporaryFile(suffix=".db", delete=False).name)
    )
    app.state.app_state = AppState(
        signal_store=signal_store,
        market_reading_assembler=assembler if with_assembler else None,
    )
    app.include_router(scan_router)
    return app


def test_scan_returns_full_match_with_met_conditions():
    # higher_tf_agrees compares the nearest higher unit's regime.trend (all bullish by
    # default), so the three XAU readings must all be present.
    readings = {
        ("XAUUSD", "M15"): _reading("XAUUSD", "M15", order_blocks=[_ob(1990, 2010)]),
        ("XAUUSD", "H1"): _reading("XAUUSD", "H1"),
        ("XAUUSD", "H4"): _reading("XAUUSD", "H4"),
    }
    store = _RecordingStore(readings)
    app = _make_app(_RecordingAssembler(store))
    client = TestClient(app)

    resp = client.post(
        "/api/conditions-scan",
        json={
            "logic": "AND",
            "conditions": [
                {"type": "higher_tf_agrees", "relation": "same"},
                {"type": "price_in_ob", "direction": "any"},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    xau = next(m for m in body["matches"] if m["timeframe"] == "M15" and m["instrument"] == "XAUUSD")
    assert xau["matched"] is True
    assert xau["met_count"] == 2
    assert xau["conditions_unmet"] == []
    assert xau["context"]["trend"] == "bullish"
    assert xau["context"]["mtf_trends"] == {"h4": "bullish", "h1": "bullish", "m15": "bullish"}
    # Every perimeter combo without a seeded reading is reported as unavailable,
    # never invented (here: all combos except the 3 seeded XAU readings).
    assert len(body["unavailable"]) == len(SCAN_COMBOS) - len(readings)
    assert all(u["reason"] == "no_reading_yet" for u in body["unavailable"])


def test_scan_reports_partial_match_transparently():
    readings = {
        ("XAUUSD", "M15"): _reading("XAUUSD", "M15"),  # no OB at price
        ("XAUUSD", "H1"): _reading("XAUUSD", "H1"),
        ("XAUUSD", "H4"): _reading("XAUUSD", "H4"),
    }
    app = _make_app(_RecordingAssembler(_RecordingStore(readings)))
    client = TestClient(app)

    resp = client.post(
        "/api/conditions-scan",
        json={
            "logic": "AND",
            "conditions": [{"type": "higher_tf_agrees", "relation": "same"}, {"type": "price_in_ob"}],
        },
    )
    assert resp.status_code == 200
    xau = next(m for m in resp.json()["matches"] if m["timeframe"] == "M15")
    assert xau["matched"] is False
    assert xau["met_count"] == 1
    assert {c["type"] for c in xau["conditions_unmet"]} == {"price_in_ob"}
    assert {c["type"] for c in xau["conditions_met"]} == {"higher_tf_agrees"}


def test_scan_is_read_only_touches_only_get_latest_reading():
    # Every perimeter combo returns a reading; writes/detection on the store raise.
    readings = {combo: _reading(combo[0], combo[1]) for combo in SCAN_COMBOS}
    store = _RecordingStore(readings)
    app = _make_app(_RecordingAssembler(store))
    client = TestClient(app)

    resp = client.post(
        "/api/conditions-scan",
        json={"logic": "OR", "conditions": [{"type": "higher_tf_agrees", "relation": "same"}]},
    )
    assert resp.status_code == 200
    # Exactly the perimeter combos read, in fixed order, and nothing else mutated.
    assert store.read_calls == list(SCAN_COMBOS)
    assert resp.json()["scanned"] == len(SCAN_COMBOS)


def test_scan_rejects_predictive_condition_type():
    app = _make_app(_RecordingAssembler(_RecordingStore({})))
    client = TestClient(app)

    resp = client.post(
        "/api/conditions-scan",
        json={"logic": "AND", "conditions": [{"type": "will_bounce"}]},
    )
    assert resp.status_code == 422  # not representable in the schema


def test_scan_requires_at_least_one_condition():
    app = _make_app(_RecordingAssembler(_RecordingStore({})))
    client = TestClient(app)
    resp = client.post("/api/conditions-scan", json={"logic": "AND", "conditions": []})
    assert resp.status_code == 422


def test_scan_503_when_assembler_not_wired():
    app = _make_app(with_assembler=False)
    client = TestClient(app)
    resp = client.post(
        "/api/conditions-scan", json={"logic": "AND", "conditions": [{"type": "higher_tf_agrees", "relation": "same"}]}
    )
    assert resp.status_code == 503


# --------------------------------------------------------------------------- #
# Freshness — reading age in candles, so the UI never asserts an aged reading
# as "présent maintenant". Descriptive only; matching is unaffected.
# --------------------------------------------------------------------------- #
_NOW = datetime(2026, 5, 28, 14, 23, tzinfo=timezone.utc)  # expected M15 close → 14:15


class TestComputeFreshness:
    def test_fresh_within_one_bar(self):
        assert _compute_freshness("M15", "2026-05-28T14:15:00+00:00", _NOW) == (0, "fresh")
        assert _compute_freshness("M15", "2026-05-28T14:00:00+00:00", _NOW) == (1, "fresh")

    def test_aging_between_two_and_four_bars(self):
        assert _compute_freshness("M15", "2026-05-28T13:15:00+00:00", _NOW) == (4, "aging")

    def test_stale_at_five_or_more_bars(self):
        assert _compute_freshness("M15", "2026-05-28T13:00:00+00:00", _NOW) == (5, "stale")

    def test_unknown_inputs_never_fabricate_staleness(self):
        assert _compute_freshness(None, "2026-05-28T14:15:00+00:00", _NOW) == (0, "fresh")
        assert _compute_freshness("M15", None, _NOW) == (0, "fresh")
        assert _compute_freshness("M15", "garbage", _NOW) == (0, "fresh")
        assert _compute_freshness("ZZ9", "2026-05-28T14:15:00+00:00", _NOW) == (0, "fresh")


def test_scan_response_carries_freshness_fields():
    # The fixtures' candle_close_ts is weeks behind real "now" → stale, with a
    # bars_behind well past the aging threshold. The fields must be present so
    # the UI can hold an aged full-match out of the "maintenant" section.
    readings = {
        ("XAUUSD", "M15"): _reading("XAUUSD", "M15"),
        ("XAUUSD", "H1"): _reading("XAUUSD", "H1"),
        ("XAUUSD", "H4"): _reading("XAUUSD", "H4"),
    }
    app = _make_app(_RecordingAssembler(_RecordingStore(readings)))
    client = TestClient(app)

    resp = client.post(
        "/api/conditions-scan",
        json={"logic": "OR", "conditions": [{"type": "higher_tf_agrees", "relation": "same"}]},
    )
    assert resp.status_code == 200
    m = next(x for x in resp.json()["matches"] if x["timeframe"] == "M15")
    assert m["freshness"] == "stale"
    assert m["bars_behind"] > 4
    # Matching itself is untouched by freshness.
    assert m["matched"] is True


def test_palette_endpoint_lists_present_tense_families_and_blocked():
    from src.intelligence.conditions_scanner import ALLOWED_CONDITION_TYPES

    app = _make_app(_RecordingAssembler(_RecordingStore({})))
    client = TestClient(app)
    resp = client.get("/api/conditions-scan/palette")
    assert resp.status_code == 200
    body = resp.json()
    palette = body["palette"]
    assert {p["type"] for p in palette} == set(ALLOWED_CONDITION_TYPES)
    assert all(p["tense"] == "present" for p in palette)
    # The interface derives its four groups + segmented buttons from the schema.
    assert body["families"] == ["structure", "zones", "liquidity", "context"]
    assert all(p["family"] in body["families"] and p["controls"] for p in palette)
    # No blocked condition left: the touch-counter mission unblocked #9, which is
    # now offered in the palette.
    assert body["blocked"] == []
    assert "zone_tested_at_most" in {p["type"] for p in palette}


def test_scan_rejects_out_of_palette_value_before_evaluation():
    # ``distribution`` is a real engine phase but is NOT offered (unreachable), so
    # a request for it is a 422 at the schema boundary — never a silent zero.
    app = _make_app(_RecordingAssembler(_RecordingStore({})))
    client = TestClient(app)
    resp = client.post(
        "/api/conditions-scan",
        json={"logic": "AND", "conditions": [{"type": "market_phase_is", "phase": "distribution"}]},
    )
    assert resp.status_code == 422


def test_scan_rejects_unknown_control_field():
    # An out-of-palette control name must be rejected, not silently ignored.
    app = _make_app(_RecordingAssembler(_RecordingStore({})))
    client = TestClient(app)
    resp = client.post(
        "/api/conditions-scan",
        json={"logic": "AND", "conditions": [{"type": "trend_is", "trend": "bullish", "will_bounce": True}]},
    )
    assert resp.status_code == 422


def test_scan_accepts_zone_tested_at_most_now_unblocked():
    # #9 is offerable since OB/FVG carry a touch count. Accepted (200), and an
    # out-of-range max_touches is still a 422 (1 ≤ N ≤ 3).
    app = _make_app(_RecordingAssembler(_RecordingStore({})))
    client = TestClient(app)
    ok = client.post(
        "/api/conditions-scan",
        json={"logic": "AND", "conditions": [{"type": "zone_tested_at_most", "max_touches": 2}]},
    )
    assert ok.status_code == 200
    bad = client.post(
        "/api/conditions-scan",
        json={"logic": "AND", "conditions": [{"type": "zone_tested_at_most", "max_touches": 5}]},
    )
    assert bad.status_code == 422


def test_scan_never_sorts_by_match_count():
    # Matches are returned in the fixed SCAN_COMBOS order regardless of how many
    # conditions each combo meets — no implicit quality ranking.
    readings = {combo: _reading(combo[0], combo[1]) for combo in SCAN_COMBOS}
    # Seed an OB only on a LATE combo so it would jump to the top under any sort.
    late = SCAN_COMBOS[-1]
    readings[late] = _reading(late[0], late[1], order_blocks=[_ob(1990, 2010)])
    app = _make_app(_RecordingAssembler(_RecordingStore(readings)))
    client = TestClient(app)
    resp = client.post(
        "/api/conditions-scan",
        json={"logic": "OR", "conditions": [{"type": "price_in_ob"}]},
    )
    assert resp.status_code == 200
    order = [(m["instrument"], m["timeframe"]) for m in resp.json()["matches"]]
    assert order == list(SCAN_COMBOS)


def test_scan_surfaces_non_evaluable_denominator():
    readings = {
        ("XAUUSD", "M15"): _reading("XAUUSD", "M15", order_blocks=[_ob(1990, 2010)]),
        ("XAUUSD", "H1"): _reading("XAUUSD", "H1"),
        ("XAUUSD", "H4"): _reading("XAUUSD", "H4"),
    }
    app = _make_app(_RecordingAssembler(_RecordingStore(readings)))
    client = TestClient(app)
    resp = client.post(
        "/api/conditions-scan",
        json={"logic": "AND", "conditions": [
            {"type": "price_in_ob"},
            {"type": "last_event_is", "event": "bos_up"},  # no events → non-evaluable
        ]},
    )
    assert resp.status_code == 200
    m = next(x for x in resp.json()["matches"] if x["timeframe"] == "M15")
    assert m["total"] == 1 and m["met_count"] == 1 and m["non_evaluable_count"] == 1
    assert len(m["conditions_unmet"]) == 0
    assert m["matched"] is True
