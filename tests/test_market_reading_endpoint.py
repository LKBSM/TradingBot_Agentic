"""Tests for the GET /api/market-reading endpoint (Chantier 2 Étape 6)."""

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import AppState
from src.api.routes.market_reading import router as market_reading_router
from src.api.signal_store import SignalStore
from src.intelligence.market_reading_schema import (
    MarketReading,
    MarketReadingConditions,
    MarketReadingEvents,
    MarketReadingHeader,
    MarketReadingRegime,
    MarketReadingStructure,
)


def _build_reading(instrument: str = "XAUUSD", timeframe: str = "M15") -> MarketReading:
    return MarketReading(
        header=MarketReadingHeader(
            instrument=instrument,
            timeframe=timeframe,
            candle_close_ts=datetime(2026, 5, 28, 14, 15, 0, tzinfo=timezone.utc),
            close_price=2378.45,
        ),
        structure=MarketReadingStructure(),
        regime=MarketReadingRegime(
            trend="bullish",
            volatility_observed="elevated",
            market_phase="expansion",
            mtf_confluence={"h1": "bullish"},
        ),
        events=MarketReadingEvents(),
        conditions=MarketReadingConditions(
            tags=["trend_bullish", "volatility_elevated"],
            description="Tendance haussière, volatilité élevée.",
            description_source="template_fallback",
        ),
    )


class _StubAssembler:
    def __init__(self, reading: MarketReading = None):
        self._reading = reading
        self.calls: list[tuple[str, str]] = []

    def get_or_generate(self, instrument: str, timeframe: str) -> MarketReading:
        self.calls.append((instrument, timeframe))
        if self._reading is not None:
            return self._reading
        return _build_reading(instrument, timeframe)


def _make_app(assembler=None, *, with_assembler: bool = True, tmp_path=None) -> FastAPI:
    """Minimal FastAPI app wired only with the market_reading router + app_state stub."""
    app = FastAPI()
    if tmp_path is not None:
        signal_store = SignalStore(db_path=str(tmp_path / "signals.db"))
    else:
        import tempfile
        signal_store = SignalStore(db_path=str(tempfile.NamedTemporaryFile(suffix=".db", delete=False).name))
    if with_assembler and assembler is None:
        assembler = _StubAssembler()
    app.state.app_state = AppState(
        signal_store=signal_store,
        market_reading_assembler=assembler if with_assembler else None,
    )
    app.include_router(market_reading_router)
    return app


def test_endpoint_returns_200_and_valid_json(tmp_path):
    app = _make_app(tmp_path=tmp_path)
    client = TestClient(app)

    resp = client.get("/api/market-reading", params={"instrument": "XAUUSD", "timeframe": "M15"})
    assert resp.status_code == 200
    body = resp.json()
    # Roundtrip through Pydantic to confirm a valid MarketReading
    reparsed = MarketReading.model_validate(body)
    assert reparsed.header.instrument == "XAUUSD"
    assert reparsed.header.timeframe == "M15"
    assert reparsed.regime.trend == "bullish"


def test_endpoint_rejects_unsupported_instrument(tmp_path):
    app = _make_app(tmp_path=tmp_path)
    client = TestClient(app)

    resp = client.get("/api/market-reading", params={"instrument": "BTCUSD", "timeframe": "M15"})
    assert resp.status_code == 400
    assert "Unsupported instrument" in resp.json()["detail"]


def test_endpoint_rejects_unsupported_timeframe(tmp_path):
    app = _make_app(tmp_path=tmp_path)
    client = TestClient(app)

    resp = client.get("/api/market-reading", params={"instrument": "XAUUSD", "timeframe": "M30"})
    assert resp.status_code == 400
    assert "Unsupported timeframe" in resp.json()["detail"]


def test_endpoint_returns_503_when_assembler_not_wired(tmp_path):
    app = _make_app(with_assembler=False, tmp_path=tmp_path)
    client = TestClient(app)

    resp = client.get("/api/market-reading", params={"instrument": "XAUUSD", "timeframe": "M15"})
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"]


def test_endpoint_returns_500_on_assembler_exception(tmp_path):
    class _BoomAssembler:
        def get_or_generate(self, instrument, timeframe):
            raise RuntimeError("boom")

    app = _make_app(assembler=_BoomAssembler(), tmp_path=tmp_path)
    client = TestClient(app)

    resp = client.get("/api/market-reading", params={"instrument": "XAUUSD", "timeframe": "M15"})
    assert resp.status_code == 500
    # Detail must NOT leak internal exception message (per existing API pattern)
    assert "boom" not in resp.json()["detail"]
    assert resp.json()["detail"] == "Internal server error"


def test_endpoint_passes_query_params_to_assembler(tmp_path):
    stub = _StubAssembler()
    app = _make_app(assembler=stub, tmp_path=tmp_path)
    client = TestClient(app)

    client.get("/api/market-reading", params={"instrument": "EURUSD", "timeframe": "H4"})
    assert stub.calls == [("EURUSD", "H4")]


def test_endpoint_missing_required_params_returns_422(tmp_path):
    app = _make_app(tmp_path=tmp_path)
    client = TestClient(app)

    resp = client.get("/api/market-reading")
    # FastAPI returns 422 (not 400) when required Query params are missing
    assert resp.status_code == 422


def test_endpoint_supports_all_three_v1_timeframes(tmp_path):
    stub = _StubAssembler()
    app = _make_app(assembler=stub, tmp_path=tmp_path)
    client = TestClient(app)

    for tf in ("M15", "H1", "H4"):
        resp = client.get("/api/market-reading", params={"instrument": "XAUUSD", "timeframe": tf})
        assert resp.status_code == 200


def test_endpoint_router_is_wired_into_app_module():
    """Smoke test that the router import path works inside src.api.app."""
    from src.api import app as app_module
    # The module imports `market_reading` from routes
    assert hasattr(app_module, "market_reading") or "market_reading" in dir(app_module)
