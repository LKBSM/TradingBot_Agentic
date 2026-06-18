"""Tests for the GET /api/live-price SSE endpoint (prototype, behind a flag).

Descriptive only: the stream carries last price + feed timestamp, nothing
predictive. A fake bridge is injected so no network/WS is involved.
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import AppState
from src.api.routes.live_price import format_price_event, router as live_price_router
from src.api.signal_store import SignalStore


class _FakeBridge:
    def __init__(self, ticks=None):
        self._ticks = ticks or {}

    def get_latest(self, instrument):
        return self._ticks.get(instrument)


def _make_app(*, bridge, tmp_path) -> FastAPI:
    app = FastAPI()
    app.state.app_state = AppState(
        signal_store=SignalStore(db_path=str(tmp_path / "signals.db")),
        live_tick_bridge=bridge,
    )
    app.include_router(live_price_router)
    return app


def test_unsupported_instrument_is_400(tmp_path):
    client = TestClient(_make_app(bridge=_FakeBridge(), tmp_path=tmp_path))
    resp = client.get("/api/live-price", params={"instrument": "BTCUSD"})
    assert resp.status_code == 400


def test_503_when_bridge_not_wired(tmp_path):
    client = TestClient(_make_app(bridge=None, tmp_path=tmp_path))
    resp = client.get("/api/live-price", params={"instrument": "XAUUSD"})
    assert resp.status_code == 503


def test_format_price_event_is_descriptive_sse_frame():
    """The wire contract: an SSE `data:` frame with ONLY {instrument, price, ts}.

    Tested on the pure formatter (not the infinite stream) so the assertion is
    deterministic and never blocks on a live generator.
    """
    frame = format_price_event("XAUUSD", 4317.5, 1781750760)
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    payload = json.loads(frame[len("data: ") :].strip())
    assert payload == {"instrument": "XAUUSD", "price": 4317.5, "ts": 1781750760}
    # Strictly descriptive — no predictive field leaks.
    for forbidden in ("forecast", "confidence_interval", "hmm_posterior", "score", "bias"):
        assert forbidden not in payload
