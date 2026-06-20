"""Conditions Scanner endpoint — read-only structural scan over the 6 combos.

POST /api/conditions-scan
  body: { "logic": "AND"|"OR", "conditions": [ {type, direction?, max_bars?}, ... ] }
  → 200 + matches[] (each with conditions_met / conditions_unmet / context) + unavailable[]
  → 422 if a condition type is not in the present-tense palette (Literal-enforced)
  → 503 if the MarketReading service is not wired

GET /api/conditions-scan/palette
  → 200 + the closed, present-tense palette (single source of truth for the builder)

DESCRIPTIVE & READ-ONLY: the scan reads the latest already-produced reading for
each combo via ``readings_store.get_latest_reading`` (a plain SELECT). It never
fetches candles, never runs detection, never writes. It cannot mutate detection.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.intelligence.conditions_scanner import (
    ALLOWED_CONDITION_TYPES,
    DEFAULT_BOS_MAX_BARS,
    PALETTE,
    evaluate_reading,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["conditions-scanner"])

# Fixed scan order — NEVER sorted by match count (no implicit quality ranking).
SCAN_COMBOS: Tuple[Tuple[str, str], ...] = (
    ("XAUUSD", "M15"),
    ("XAUUSD", "H1"),
    ("XAUUSD", "H4"),
    ("EURUSD", "M15"),
    ("EURUSD", "H1"),
    ("EURUSD", "H4"),
)

# Present-tense condition types ONLY. Drift from the palette is caught below.
ConditionType = Literal[
    "mtf_aligned",
    "price_in_ob",
    "price_in_fvg",
    "ob_fvg_confluence",
    "bos_recent_confirmed",
]
DirectionFilter = Literal["any", "bullish", "bearish"]

# Guard: the request-model vocabulary must equal the engine palette exactly, so a
# predictive type can never slip in on one side only.
assert set(ConditionType.__args__) == set(ALLOWED_CONDITION_TYPES), (  # noqa: S101
    "ConditionType Literal drifted from conditions_scanner.PALETTE"
)


# ── Request models ──────────────────────────────────────────────────────────


class ScanCondition(BaseModel):
    type: ConditionType
    direction: DirectionFilter = "any"
    max_bars: int = Field(default=DEFAULT_BOS_MAX_BARS, ge=1, le=50)


class ConditionsScanRequest(BaseModel):
    logic: Literal["AND", "OR"] = "AND"
    conditions: List[ScanCondition] = Field(..., min_length=1, max_length=10)


# ── Response models ─────────────────────────────────────────────────────────


class ConditionOutcome(BaseModel):
    type: str
    label: str
    met: bool
    detail: str


class ComboMatch(BaseModel):
    instrument: Optional[str]
    timeframe: Optional[str]
    candle_close_ts: Optional[str]
    close_price: Optional[float]
    matched: bool
    met_count: int
    total: int
    conditions_met: List[ConditionOutcome]
    conditions_unmet: List[ConditionOutcome]
    context: Dict[str, Any]


class UnavailableCombo(BaseModel):
    instrument: str
    timeframe: str
    reason: str


class ConditionsScanResponse(BaseModel):
    as_of: str
    logic: str
    scanned: int
    matches: List[ComboMatch]
    unavailable: List[UnavailableCombo]


class PaletteResponse(BaseModel):
    palette: List[Dict[str, Any]]


# ── Routes ──────────────────────────────────────────────────────────────────


@router.get("/conditions-scan/palette", response_model=PaletteResponse)
async def get_palette() -> PaletteResponse:
    """The closed, present-tense palette — single source of truth for the builder."""
    return PaletteResponse(palette=PALETTE)


@router.post("/conditions-scan", response_model=ConditionsScanResponse)
async def conditions_scan(request: Request, body: ConditionsScanRequest) -> ConditionsScanResponse:
    assembler = getattr(request.app.state.app_state, "market_reading_assembler", None)
    if assembler is None:
        raise HTTPException(status_code=503, detail="MarketReading service not configured")
    store = getattr(assembler, "readings_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Readings store not available")

    conditions = [c.model_dump() for c in body.conditions]
    matches: List[ComboMatch] = []
    unavailable: List[UnavailableCombo] = []

    for instrument, timeframe in SCAN_COMBOS:
        # READ-ONLY: plain SELECT of the latest already-produced reading.
        reading = store.get_latest_reading(instrument, timeframe)
        if reading is None:
            unavailable.append(
                UnavailableCombo(instrument=instrument, timeframe=timeframe, reason="no_reading_yet")
            )
            continue
        try:
            combo = evaluate_reading(reading, conditions, body.logic)
        except Exception:
            logger.exception("conditions-scan eval failed for %s/%s", instrument, timeframe)
            unavailable.append(
                UnavailableCombo(
                    instrument=instrument, timeframe=timeframe, reason="evaluation_error"
                )
            )
            continue
        matches.append(ComboMatch(**combo))

    return ConditionsScanResponse(
        as_of=datetime.now(timezone.utc).isoformat(),
        logic=body.logic,
        scanned=len(SCAN_COMBOS),
        matches=matches,
        unavailable=unavailable,
    )
