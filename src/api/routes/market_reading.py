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

from src.api.entitlements import enforce_combo_access
from src.api.session_auth import optional_account
from src.intelligence.market_reading_schema import MarketReading

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["market-reading"])

# V1 perimeter per docs/architecture/MIA_MARKETS_V2_VISION.md §1.3
SUPPORTED_INSTRUMENTS = frozenset({"XAUUSD", "EURUSD"})
SUPPORTED_TIMEFRAMES = frozenset({"M15", "H1", "H4"})


@router.get("/market-reading", response_model=MarketReading)
async def get_market_reading(
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

    # Freemium gate (no-op while the gate is OFF): free tier sees XAU/USD M15.
    enforce_combo_access(request, account, instrument, timeframe)

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
    except Exception:
        logger.exception("market-reading generation failed for %s/%s", instrument, timeframe)
        raise HTTPException(status_code=500, detail="Internal server error")
