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

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.entitlements import enforce_scanner_access
from src.api.session_auth import optional_account
from src.intelligence.conditions_scanner import (
    ALLOWED_CONDITION_TYPES,
    DEFAULT_BOS_MAX_BARS,
    DEFAULT_PROXIMITY_PCT,
    PALETTE,
    evaluate_reading,
)
from src.intelligence.market_reading_assembler import expected_last_candle_close

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

# Candle duration per timeframe — used to express a reading's age in *bars*
# (market-cadence units) rather than wall-clock, so freshness is comparable
# across timeframes. Factual; no prediction.
_TF_MINUTES: Dict[str, int] = {"M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}

# Freshness tiers, in bars behind the latest expected close. A healthy combo
# (the 60s scheduler keeps the scanner perimeter warm) sits at 0–1 bar.
FreshnessState = Literal["fresh", "aging", "stale"]
_FRESH_MAX_BARS = 1   # 0–1 bar behind  → up to date
_AGING_MAX_BARS = 4   # 2–4 bars behind → starting to date; ≥5 → stale


def _compute_freshness(
    timeframe: Optional[str], candle_close_ts: Optional[str], now: datetime
) -> Tuple[int, FreshnessState]:
    """How many closed candles a reading is behind the latest expected close.

    Purely descriptive: compares the reading's ``candle_close_ts`` to the most
    recent boundary that has elapsed at ``now`` (same clock the scheduler uses).
    Returns ``(bars_behind, state)``. Unknown/unparsable inputs return
    ``(0, "fresh")`` — we never fabricate staleness we cannot prove.
    """
    if not timeframe or not candle_close_ts:
        return 0, "fresh"
    tf_minutes = _TF_MINUTES.get(timeframe.upper())
    if not tf_minutes:
        return 0, "fresh"
    try:
        expected = expected_last_candle_close(timeframe, now)
        stored = datetime.fromisoformat(str(candle_close_ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return 0, "fresh"
    if stored.tzinfo is None:
        stored = stored.replace(tzinfo=timezone.utc)
    delta_minutes = (expected - stored.astimezone(timezone.utc)).total_seconds() / 60.0
    bars_behind = max(0, round(delta_minutes / tf_minutes))
    if bars_behind <= _FRESH_MAX_BARS:
        return bars_behind, "fresh"
    if bars_behind <= _AGING_MAX_BARS:
        return bars_behind, "aging"
    return bars_behind, "stale"


# Present-tense condition types ONLY. Drift from the palette is caught below.
ConditionType = Literal[
    "mtf_aligned",
    "trend_is",
    "market_phase_is",
    "volatility_is",
    "price_in_ob",
    "price_in_fvg",
    "ob_fvg_confluence",
    "bos_recent_confirmed",
    "choch_recent_confirmed",
    "retest_in_progress",
    "price_near_ob",
    "price_near_fvg",
    "price_near_liquidity",
    "liquidity_swept_recent",
]
DirectionFilter = Literal["any", "bullish", "bearish"]
LiquiditySideFilter = Literal["any", "bsl", "ssl"]
TrendChoice = Literal["bullish", "bearish", "ranging", "neutral"]
PhaseChoice = Literal["accumulation", "distribution", "trend", "ranging", "expansion"]
VolatilityChoice = Literal["low", "normal", "elevated"]

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
    # Regime selectors (used by trend_is / market_phase_is / volatility_is).
    trend: Optional[TrendChoice] = None
    phase: Optional[PhaseChoice] = None
    volatility: Optional[VolatilityChoice] = None
    # Proximity threshold (% of price) for the "price near …" conditions.
    proximity_pct: float = Field(default=DEFAULT_PROXIMITY_PCT, gt=0, le=10)
    # Liquidity-side filter for the liquidity conditions (bsl above / ssl below).
    side: LiquiditySideFilter = "any"


class ConditionsScanRequest(BaseModel):
    logic: Literal["AND", "OR"] = "AND"
    conditions: List[ScanCondition] = Field(..., min_length=1, max_length=10)


# ── Response models ─────────────────────────────────────────────────────────


class ConditionOutcome(BaseModel):
    type: str
    label: str
    met: bool
    # False when the data needed to judge this condition is missing (e.g. a
    # sibling timeframe has no reading yet) — distinct from "judged and not met".
    available: bool = True
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
    # Reading age in candles behind the latest expected close, and its tier.
    # Lets the UI avoid asserting an aged reading as "présent maintenant".
    bars_behind: int = 0
    freshness: FreshnessState = "fresh"


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
async def conditions_scan(
    request: Request,
    body: ConditionsScanRequest,
    account: Optional[Dict[str, Any]] = Depends(optional_account),
) -> ConditionsScanResponse:
    # Freemium gate (no-op while the gate is OFF): the multi-market scanner is a
    # paid feature — a free account is invited to subscribe (402), never errored.
    enforce_scanner_access(request, account)

    assembler = getattr(request.app.state.app_state, "market_reading_assembler", None)
    if assembler is None:
        raise HTTPException(status_code=503, detail="MarketReading service not configured")
    store = getattr(assembler, "readings_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Readings store not available")

    now = datetime.now(timezone.utc)
    conditions = [c.model_dump() for c in body.conditions]
    matches: List[ComboMatch] = []
    unavailable: List[UnavailableCombo] = []

    # Load every combo's latest reading ONCE (read-only SELECTs), then build a
    # per-instrument {timeframe → regime.trend} map so mtf_aligned can judge
    # cross-timeframe alignment from each timeframe's OWN trend — the same source
    # the chart's "Régime" panel uses — instead of the structurally-incomplete
    # per-reading mtf_confluence field.
    readings: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
    trends_by_instrument: Dict[str, Dict[str, Optional[str]]] = {}
    for instrument, timeframe in SCAN_COMBOS:
        reading = store.get_latest_reading(instrument, timeframe)
        readings[(instrument, timeframe)] = reading
        if reading is not None:
            trend = (reading.get("regime") or {}).get("trend")
            trends_by_instrument.setdefault(instrument, {})[timeframe] = trend

    for instrument, timeframe in SCAN_COMBOS:
        reading = readings[(instrument, timeframe)]
        if reading is None:
            unavailable.append(
                UnavailableCombo(instrument=instrument, timeframe=timeframe, reason="no_reading_yet")
            )
            continue
        try:
            combo = evaluate_reading(
                reading, conditions, body.logic, trends_by_instrument.get(instrument, {})
            )
        except Exception:
            logger.exception("conditions-scan eval failed for %s/%s", instrument, timeframe)
            unavailable.append(
                UnavailableCombo(
                    instrument=instrument, timeframe=timeframe, reason="evaluation_error"
                )
            )
            continue
        bars_behind, freshness = _compute_freshness(
            timeframe, combo.get("candle_close_ts"), now
        )
        combo["bars_behind"] = bars_behind
        combo["freshness"] = freshness
        matches.append(ComboMatch(**combo))

    return ConditionsScanResponse(
        as_of=now.isoformat(),
        logic=body.logic,
        scanned=len(SCAN_COMBOS),
        matches=matches,
        unavailable=unavailable,
    )
