"""Tests for POST /api/conditions-scan (read-only structural scan).

Covers: correct met/unmet over combos, partial (transparency) results, the
read-only guarantee (only get_latest_reading is touched — never a write nor
detection), predictive types rejected at the schema boundary, and 503 wiring.
"""

from __future__ import annotations

import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import AppState
from src.api.routes.conditions_scan import SCAN_COMBOS, router as scan_router
from src.api.signal_store import SignalStore


def _reading(instrument, timeframe, *, close_price=2000.0, mtf=None, order_blocks=None, bos=None):
    return {
        "header": {
            "instrument": instrument,
            "timeframe": timeframe,
            "candle_close_ts": "2026-05-28T14:15:00+00:00",
            "close_price": close_price,
        },
        "structure": {
            "bos": bos,
            "choch": None,
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
        "conditions": {"tags": [], "description": "", "description_source": "template_fallback"},
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
    # mtf_aligned is judged from each timeframe's OWN regime.trend (all bullish by
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
                {"type": "mtf_aligned", "direction": "bullish"},
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
    # The 3 EURUSD combos have no reading → reported as unavailable, never invented.
    assert len(body["unavailable"]) == 3


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
            "conditions": [{"type": "mtf_aligned"}, {"type": "price_in_ob"}],
        },
    )
    assert resp.status_code == 200
    xau = next(m for m in resp.json()["matches"] if m["timeframe"] == "M15")
    assert xau["matched"] is False
    assert xau["met_count"] == 1
    assert {c["type"] for c in xau["conditions_unmet"]} == {"price_in_ob"}
    assert {c["type"] for c in xau["conditions_met"]} == {"mtf_aligned"}


def test_scan_is_read_only_touches_only_get_latest_reading():
    # All 6 combos return a reading; writes/detection on the store raise.
    readings = {combo: _reading(combo[0], combo[1]) for combo in SCAN_COMBOS}
    store = _RecordingStore(readings)
    app = _make_app(_RecordingAssembler(store))
    client = TestClient(app)

    resp = client.post(
        "/api/conditions-scan",
        json={"logic": "OR", "conditions": [{"type": "mtf_aligned"}]},
    )
    assert resp.status_code == 200
    # Exactly the 6 combos read, in fixed order, and nothing else mutated.
    assert store.read_calls == list(SCAN_COMBOS)
    assert resp.json()["scanned"] == 6


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
        "/api/conditions-scan", json={"logic": "AND", "conditions": [{"type": "mtf_aligned"}]}
    )
    assert resp.status_code == 503


def test_palette_endpoint_lists_present_tense_only():
    app = _make_app(_RecordingAssembler(_RecordingStore({})))
    client = TestClient(app)
    resp = client.get("/api/conditions-scan/palette")
    assert resp.status_code == 200
    palette = resp.json()["palette"]
    assert len(palette) == 10
    assert all(p["tense"] == "present" for p in palette)
