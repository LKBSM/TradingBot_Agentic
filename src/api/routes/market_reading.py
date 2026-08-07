"""Market Reading endpoint — Chantier 2 lazy on-demand mode.

GET /api/market-reading?instrument=XAUUSD&timeframe=M15
  → 200 + MarketReading JSON
  → 400 if instrument/timeframe not in V1 perimeter
  → 503 if assembler not wired in app_state
"""

from __future__ import annotations

import logging

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.api.subscription_gate import enforce_access
from src.api.session_auth import optional_account
from src.intelligence.market_reading_assembler import MarketReadingDataUnavailable
from src.intelligence.market_reading_schema import MarketReading

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["market-reading"])

# Perimeter — single source of truth = the LB-1 lookback config (M1..D1). A
# request for a combo with no cached data simply 404s downstream; validation
# accepts the full supported set so an M1 deep-link works once the gate is on.
from src.intelligence.lookback_config import supported_instruments, supported_timeframes

SUPPORTED_INSTRUMENTS = frozenset(supported_instruments())
SUPPORTED_TIMEFRAMES = frozenset(supported_timeframes())


# REC-1: SYNC endpoint on purpose. ``get_or_generate`` does blocking work
# (detection + provider fetch, ~1.7s). As ``async def`` it ran ON the event loop
# and froze EVERY other request — including the SubscriptionGate's
# ``/api/access/me`` — for its whole duration, which left all pages stuck on the
# loading skeleton. FastAPI runs a plain ``def`` path operation in a worker
# thread, so a slow reading no longer blocks the loop. No behaviour/data change.
@router.get("/market-reading", response_model=MarketReading)
def get_market_reading(
    request: Request,
    instrument: str = Query(..., description="XAUUSD or EURUSD"),
    timeframe: str = Query(..., description="M15, H1, or H4"),
    account: Optional[Dict[str, Any]] = Depends(optional_account),
) -> MarketReading:
    if instrument not in SUPPORTED_INSTRUMENTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported instrument '{instrument}'. "
                f"Supported: {sorted(SUPPORTED_INSTRUMENTS)}"
            ),
        )
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {sorted(SUPPORTED_TIMEFRAMES)}"
            ),
        )

    # Paid-only gate (no-op while the gate is OFF): an unsubscribed account gets
    # no market data at all — not even one candle (PAY-1). Owner/subscriber pass.
    enforce_access(request, account)

    assembler = getattr(request.app.state.app_state, "market_reading_assembler", None)
    if assembler is None:
        raise HTTPException(
            status_code=503,
            detail="MarketReading service not configured",
        )

    try:
        return assembler.get_or_generate(instrument, timeframe)
    except HTTPException:
        raise
    except MarketReadingDataUnavailable as exc:
        # Valid combo, but no data anywhere (feed down AND cache empty AND nothing
        # stored). Distinct 404 so the UI says « aucune donnée pour cette
        # combinaison » rather than a generic failure. Not an internal bug.
        logger.warning("market-reading: no data for %s/%s: %s", instrument, timeframe, exc)
        raise HTTPException(
            status_code=404,
            detail="No market data available for this instrument/timeframe yet.",
        )
    except Exception:
        logger.exception("market-reading generation failed for %s/%s", instrument, timeframe)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/market-status")
async def get_market_status(
    request: Request,
    instrument: str = Query(..., description="XAUUSD or EURUSD"),
    timeframe: str = Query(..., description="M15, H1, or H4"),
) -> Dict[str, Any]:
    """Server-side market status for (instrument, timeframe) — the single source
    of truth the App badge, Scanner and M.I.A agent all read (never the client
    clock). Fact-first: the last closed candle governs; the calendar names the
    reason and gives the reopen time. A pure read — never fetches or rebuilds."""
    if instrument not in SUPPORTED_INSTRUMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported instrument '{instrument}'. Supported: {sorted(SUPPORTED_INSTRUMENTS)}",
        )
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported timeframe '{timeframe}'. Supported: {sorted(SUPPORTED_TIMEFRAMES)}",
        )
    assembler = getattr(request.app.state.app_state, "market_reading_assembler", None)
    if assembler is None:
        raise HTTPException(status_code=503, detail="MarketReading service not configured")
    try:
        return assembler.market_status(instrument, timeframe).to_dict()
    except Exception:
        logger.exception("market-status failed for %s/%s", instrument, timeframe)
        raise HTTPException(status_code=500, detail="Internal server error")
