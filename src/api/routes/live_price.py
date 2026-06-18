"""Live last-price SSE endpoint (PROTOTYPE — dev / free tier, behind a flag).

GET /api/live-price?instrument=XAUUSD
  → text/event-stream of {instrument, price, ts} JSON frames (descriptive only)
  → 400 if the instrument is outside the V1 perimeter
  → 503 if the live-tick bridge is not wired (LIVE_TICK_ENABLED off)

WHY this exists: the default app refreshes zone interaction only at candle close.
This stream feeds an OPT-IN live overlay that updates the INTERACTION of already-
detected zones intra-candle (FVG shrinking as price fills it, OB shown "in test").

STRICT SCOPE:
  - Descriptive only — the payload is the last traded price + its feed timestamp.
    No forecast, no structure, no BOS/CHOCH, no predictive field whatsoever.
  - One shared backend WS connection feeds all SSE clients (free-trial 1-conn
    limit). The browser never opens a WS and never sees the API key.
  - Additive: when LIVE_TICK_ENABLED is off, this route returns 503 and the app
    behaves exactly as before (candle-close refresh).
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["live-price"])

# V1 perimeter — mirrors candles.py / market_reading.py.
SUPPORTED_INSTRUMENTS = frozenset({"XAUUSD", "EURUSD"})

# How often the generator polls the bridge for a fresh tick, and how often it
# emits a keepalive comment when the price hasn't moved (keeps proxies from
# closing an idle stream). Both small enough to feel "live", cheap (a dict read).
_POLL_INTERVAL_S = 0.5
_KEEPALIVE_EVERY_S = 15.0


def format_price_event(instrument: str, price: float, ts: int) -> str:
    """Serialise one tick as an SSE ``data:`` frame (descriptive only).

    Extracted so the wire contract is unit-testable without driving the infinite
    stream. The payload is strictly {instrument, price, ts} — no predictive field.
    """
    payload = json.dumps({"instrument": instrument, "price": price, "ts": ts})
    return f"data: {payload}\n\n"


@router.get("/live-price")
async def live_price(
    request: Request,
    instrument: str = Query(..., description="XAUUSD or EURUSD"),
) -> StreamingResponse:
    if instrument not in SUPPORTED_INSTRUMENTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported instrument '{instrument}'. "
                f"Supported: {sorted(SUPPORTED_INSTRUMENTS)}"
            ),
        )

    bridge = _resolve_bridge(request)
    if bridge is None:
        # LIVE_TICK_ENABLED is off (default) — live overlay simply unavailable.
        raise HTTPException(status_code=503, detail="Live tick bridge not configured")

    async def event_stream():
        last_emitted_ts: int | None = None
        last_emitted_price: float | None = None
        since_keepalive = 0.0
        try:
            while True:
                if await request.is_disconnected():
                    break
                tick = bridge.get_latest(instrument)
                if tick is not None and (
                    tick.ts != last_emitted_ts or tick.price != last_emitted_price
                ):
                    last_emitted_ts = tick.ts
                    last_emitted_price = tick.price
                    since_keepalive = 0.0
                    yield format_price_event(instrument, tick.price, tick.ts)
                else:
                    since_keepalive += _POLL_INTERVAL_S
                    if since_keepalive >= _KEEPALIVE_EVERY_S:
                        since_keepalive = 0.0
                        yield ": keepalive\n\n"
                await asyncio.sleep(_POLL_INTERVAL_S)
        except asyncio.CancelledError:  # pragma: no cover — client/server teardown
            raise

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering for SSE
        },
    )


def _resolve_bridge(request: Request):
    """Return the shared live-tick bridge from app_state, or None if absent."""
    app_state = getattr(request.app.state, "app_state", None)
    if app_state is None:
        return None
    return getattr(app_state, "live_tick_bridge", None)
