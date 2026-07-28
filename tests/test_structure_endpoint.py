"""LB-1 — /api/coverage (honest depth) + /api/structure (window-bounded)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import AppState
from src.api.routes.structure import router as structure_router
from src.api.signal_store import SignalStore
from src.intelligence import lookback_config
from src.intelligence.data_providers.twelve_data_provider import Candle
from src.storage import CandlesCacheStore
from src.storage.structure_store import StructureStore

UTC = timezone.utc
T0 = datetime(2026, 6, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_m1(monkeypatch):
    monkeypatch.delenv("LB1_ENABLE_M1", raising=False)
    monkeypatch.delenv("SENTINEL_LOOKBACK_DEPTHS_PATH", raising=False)
    lookback_config.reset_cache()
    yield
    lookback_config.reset_cache()


class _StubAssembler:
    def __init__(self, store):
        self.candles_store = store


def _make_app(tmp_path, *, seed_candles=True, structure_store=None):
    app = FastAPI()
    candles_store = CandlesCacheStore(db_path=str(tmp_path / "candles.db"))
    if seed_candles:
        candles_store.upsert_candles("XAUUSD", "M15", [
            Candle(ts=T0 + timedelta(minutes=15 * i), open=2000, high=2001, low=1999, close=2000.5, volume=1.0)
            for i in range(50)
        ])
    state = AppState(signal_store=SignalStore(db_path=str(tmp_path / "s.db")),
                     market_reading_assembler=_StubAssembler(candles_store))
    if structure_store is not None:
        state.structure_store = structure_store
    app.state.app_state = state
    app.include_router(structure_router)
    return app


def test_coverage_lists_all_enabled_combos(tmp_path):
    client = TestClient(_make_app(tmp_path))
    body = client.get("/api/coverage").json()
    combos = body["combos"]
    assert len(combos) == len(lookback_config.enabled_combos()) == 10
    xau_m15 = next(c for c in combos if c["instrument"] == "XAUUSD" and c["timeframe"] == "M15")
    assert xau_m15["candle_count"] == 50
    assert xau_m15["history_since"] is not None
    assert xau_m15["duration"] == "1mo"


def test_coverage_reports_partial_honestly(tmp_path):
    """A combo with far fewer candles than its target is NOT marked complete."""
    client = TestClient(_make_app(tmp_path))
    body = client.get("/api/coverage").json()
    xau_m15 = next(c for c in body["combos"] if c["timeframe"] == "M15" and c["instrument"] == "XAUUSD")
    # 50 candles vs a ~2900 target → incomplete, and history_since is real.
    assert xau_m15["candle_count"] < xau_m15["target_bars"]
    assert xau_m15["complete"] is False
    # An untouched combo reports empty, never a fabricated start.
    empty = next(c for c in body["combos"] if c["timeframe"] == "H4" and c["instrument"] == "EURUSD")
    assert empty["candle_count"] == 0 and empty["history_since"] is None and empty["complete"] is False


def _seed_structure(store):
    zones = {"order_blocks": [
        {"direction": "bullish", "level_high": 2010, "level_low": 2000,
         "created_at": T0 + timedelta(hours=1), "status": "active", "tested": False, "mitigated_at": None},
        {"direction": "bearish", "level_high": 2210, "level_low": 2200,
         "created_at": T0 + timedelta(hours=2), "status": "active", "tested": True, "mitigated_at": None},
    ], "fair_value_gaps": []}
    events = {"bos_events": [
        {"direction": "bullish", "level": 2005, "broken_at": T0 + timedelta(hours=1)},
    ], "choch_events": []}
    store.apply_snapshot("XAUUSD", "H1", zones, events, window_start=T0, processed_ts=T0 + timedelta(hours=3))


def test_structure_window_filters_by_price(tmp_path):
    store = StructureStore(db_path=str(tmp_path / "struct.db"))
    _seed_structure(store)
    client = TestClient(_make_app(tmp_path, structure_store=store))

    # Narrow price band around 2000 → only the bullish zone; totals show 2.
    resp = client.get("/api/structure", params={
        "instrument": "XAUUSD", "timeframe": "H1", "price_low": 1995, "price_high": 2015,
    })
    body = resp.json()
    assert len(body["zones"]) == 1
    assert body["zones"][0]["direction"] == "bullish"
    assert body["zones_total"] == 2   # denominator for "N sur M"
    assert body["events_total"] == 1


def test_structure_window_filters_by_time(tmp_path):
    store = StructureStore(db_path=str(tmp_path / "struct.db"))
    _seed_structure(store)
    client = TestClient(_make_app(tmp_path, structure_store=store))
    # Only events at/after T0+90min → excludes the single event at +1h.
    resp = client.get("/api/structure", params={
        "instrument": "XAUUSD", "timeframe": "H1",
        "time_from": (T0 + timedelta(minutes=90)).isoformat(),
    })
    assert resp.json()["events"] == []


def test_structure_rejects_unsupported_instrument(tmp_path):
    store = StructureStore(db_path=str(tmp_path / "struct.db"))
    client = TestClient(_make_app(tmp_path, structure_store=store))
    resp = client.get("/api/structure", params={"instrument": "NOPE", "timeframe": "H1"})
    assert resp.status_code == 400
