"""Tests for the GET /api/candles endpoint (live-data wiring).

The endpoint is STRICTLY DESCRIPTIVE: it serves only OHLC + UTC epoch timestamps
read from candles_cache. These tests pin that contract and assert that no
predictive (InsightSignalV2) field ever leaks into the response.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import AppState
from src.api.routes.candles import router as candles_router
from src.api.signal_store import SignalStore
from src.intelligence.data_providers.twelve_data_provider import Candle
from src.storage import CandlesCacheStore

# Fields that belong to InsightSignalV2 and must NEVER appear in /api/candles.
FORBIDDEN_PREDICTIVE_KEYS = frozenset(
    {
        "forecast",
        "forecast_atr_pips",
        "confidence_interval",
        "conformal",
        "hmm_posterior",
        "bocpd_changepoint_prob",
        "target_1",
        "target_2",
        "valid_until",
        "valid_until_utc",
        "confluence_signal",
        "confluence_score",
        "score",
        "bias",
        "direction",
    }
)


class _StubAssembler:
    """Mimics the assembler exposing a populated candles_store property."""

    def __init__(self, store):
        self.candles_store = store


def _candle(seconds: int, close: float) -> Candle:
    return Candle(
        ts=datetime(2026, 5, 29, 14, 0, seconds, tzinfo=timezone.utc),
        open=close - 1.0,
        high=close + 2.0,
        low=close - 2.0,
        close=close,
        volume=100.0 + seconds,
    )


def _make_app(*, with_store: bool = True, tmp_path, seed=True) -> FastAPI:
    app = FastAPI()
    signal_store = SignalStore(db_path=str(tmp_path / "signals.db"))
    assembler = None
    if with_store:
        candles_store = CandlesCacheStore(db_path=str(tmp_path / "candles.db"))
        if seed:
            candles_store.upsert_candles(
                "XAUUSD",
                "M15",
                [_candle(i, 2378.0 + i) for i in range(5)],
            )
        assembler = _StubAssembler(candles_store)
    app.state.app_state = AppState(
        signal_store=signal_store,
        market_reading_assembler=assembler,
    )
    app.include_router(candles_router)
    return app


def test_returns_n_well_formed_candles(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles", params={"instrument": "XAUUSD", "timeframe": "M15"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["instrument"] == "XAUUSD"
    assert body["timeframe"] == "M15"
    candles = body["candles"]
    assert len(candles) == 5
    # Each candle has exactly the descriptive OHLC keys, nothing more.
    for c in candles:
        assert set(c.keys()) == {"time", "open", "high", "low", "close", "volume"}
        assert isinstance(c["time"], int)
        assert c["high"] >= c["low"]
    # Ascending by time (chart expects oldest-first; never a future projection).
    times = [c["time"] for c in candles]
    assert times == sorted(times)


def test_time_is_utc_epoch_seconds(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles", params={"instrument": "XAUUSD", "timeframe": "M15"}
    )
    first = resp.json()["candles"][0]
    expected = int(datetime(2026, 5, 29, 14, 0, 0, tzinfo=timezone.utc).timestamp())
    assert first["time"] == expected


def test_limit_caps_window(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles",
        params={"instrument": "XAUUSD", "timeframe": "M15", "limit": 2},
    )
    assert resp.status_code == 200
    candles = resp.json()["candles"]
    assert len(candles) == 2
    # Most recent two, still ascending.
    assert [c["close"] for c in candles] == [2381.0, 2382.0]


def test_no_predictive_field_leaks(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles", params={"instrument": "XAUUSD", "timeframe": "M15"}
    )
    raw = resp.text.lower()
    for key in FORBIDDEN_PREDICTIVE_KEYS:
        assert key.lower() not in raw, f"predictive field '{key}' leaked into /api/candles"


def test_rejects_unsupported_instrument(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles", params={"instrument": "BTCUSD", "timeframe": "M15"}
    )
    assert resp.status_code == 400
    assert "Unsupported instrument" in resp.json()["detail"]


def test_rejects_unsupported_timeframe(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles", params={"instrument": "XAUUSD", "timeframe": "M30"}
    )
    assert resp.status_code == 400
    assert "Unsupported timeframe" in resp.json()["detail"]


def test_404_when_combo_has_no_cached_candles(tmp_path):
    # Valid combo (EURUSD/H4) but nothing seeded for it.
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles", params={"instrument": "EURUSD", "timeframe": "H4"}
    )
    assert resp.status_code == 404
    assert "No candles cached" in resp.json()["detail"]


def test_503_when_store_not_wired(tmp_path):
    client = TestClient(_make_app(with_store=False, tmp_path=tmp_path))
    resp = client.get(
        "/api/candles", params={"instrument": "XAUUSD", "timeframe": "M15"}
    )
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"]


def test_limit_out_of_range_returns_422(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles",
        params={"instrument": "XAUUSD", "timeframe": "M15", "limit": 0},
    )
    assert resp.status_code == 422


def test_router_wired_into_app_module():
    from src.api import app as app_module

    assert hasattr(app_module, "candles") or "candles" in dir(app_module)
