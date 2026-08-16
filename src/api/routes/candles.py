"""OHLC candles endpoint — descriptive chart feed for the /app view.

GET /api/candles?instrument=XAUUSD&timeframe=M15&limit=200
  → 200 + {instrument, timeframe, candles: [{time, open, high, low, close, volume}]}
  → 400 if instrument/timeframe not in V1 perimeter
  → 404 if the combo is in perimeter but has no cached candles yet
  → 503 if the candles store is not wired in app_state

STRICTLY DESCRIPTIVE: this endpoint exposes only raw OHLC + UTC timestamps read
from the ``candles_cache`` SQLite table (already populated by the bootstrap +
scheduler). It NEVER returns any predictive field (no forecast, no conformal
interval, no hmm_posterior, no confluence score) — those live in InsightSignalV2,
which is not served here. The series stops at the last fully-closed candle; there
is no forward projection.

No external provider (Twelve Data) call is made here — pure cache read, so the
free-tier budget is untouched by chart fetches.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from src.api.subscription_gate import enforce_access
from src.api.session_auth import optional_account

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["candles"])

# Perimeter — single source of truth = the LB-1 lookback config (M1..D1), plus
# W1 as a read-only reference series. M1..H4/D1 feed the chart; D1/W1 also derive
# the calendar reference levels of the Régime panel — day/week open, previous
# day/week extremes. The "day"/"week" boundary is therefore the DATA FEED's own
# D1/W1 candle (single definition, shared with the chart), never a second
# client-side boundary. See RG-1 / MC-1 audits.
from src.intelligence.lookback_config import supported_instruments, supported_timeframes

SUPPORTED_INSTRUMENTS = frozenset(supported_instruments())
SUPPORTED_TIMEFRAMES = frozenset(supported_timeframes()) | {"W1"}
# Reference series for the Régime « Niveaux de référence » tile. Not kept warm by
# the scheduler → populated on demand from the feed on a cache-miss (see below).
REFERENCE_TIMEFRAMES = frozenset({"D1", "W1"})

# Default / max window. Widened 2026-06-15 alongside the assembler lookback
# (now 500) so the chart can render indicator-grade history. Payload stays small
# (OHLC only, gzipped). Never serve more than the assembler caches.
DEFAULT_LIMIT = 300
MAX_LIMIT = 1000


class CandleOut(BaseModel):
    """A single OHLC candle. ``time`` is a UTC epoch in SECONDS (UTCTimestamp).

    Descriptive only — no derived/predictive fields.
    """

    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandlesResponse(BaseModel):
    """Envelope for the candle window served to the chart."""

    instrument: str
    timeframe: str
    candles: List[CandleOut]
    # CHART-2 — true when the cache holds candles OLDER than the oldest one
    # returned here, i.e. a further backward page exists. The chart uses it to
    # know when it has reached the real START of available history (so it can say
    # so honestly instead of hitting an invisible wall). Defaults False so older
    # payloads / callers that ignore paging keep working unchanged.
    has_more_history: bool = False


@router.get("/candles", response_model=CandlesResponse)
async def get_candles(
    request: Request,
    instrument: str = Query(..., description="XAUUSD or EURUSD"),
    timeframe: str = Query(..., description="M15, H1, H4 (chart) or D1, W1 (reference levels)"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Max candles"),
    before: Optional[int] = Query(
        None,
        ge=0,
        description=(
            "CHART-2 backward paging: UTC epoch SECONDS. When set, return up to "
            "`limit` candles strictly OLDER than this timestamp (the chart's "
            "on-demand history load), instead of the most recent window."
        ),
    ),
    account: Optional[Dict[str, Any]] = Depends(optional_account),
) -> CandlesResponse:
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

    # Paid-only gate (no-op while the gate is OFF): no candle without an active
    # subscription (PAY-1). Owner/subscriber pass.
    enforce_access(request, account)

    store = _resolve_candles_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Candles store not configured")

    try:
        if before is not None:
            cutoff = datetime.fromtimestamp(before, tz=timezone.utc)
            candles = store.get_candles_before(instrument, timeframe, cutoff, limit)
        else:
            candles = store.get_last_n_candles(instrument, timeframe, limit)
    except Exception:
        logger.exception("candles read failed for %s/%s", instrument, timeframe)
        raise HTTPException(status_code=500, detail="Internal server error")

    # D1/W1 are REFERENCE series (Régime « Niveaux de référence » tile), not kept
    # warm by the scheduler. Populate them on demand from the feed — one provider
    # call per closed daily/weekly bar (warm_candles is market-aware + idempotent).
    # Only on the initial (most-recent) read; a backward page never warms.
    if not candles and before is None and timeframe in REFERENCE_TIMEFRAMES:
        assembler = _resolve_assembler(request)
        if assembler is not None and hasattr(assembler, "warm_candles"):
            if assembler.warm_candles(instrument, timeframe) > 0:
                try:
                    candles = store.get_last_n_candles(instrument, timeframe, limit)
                except Exception:
                    logger.exception("candles re-read failed for %s/%s", instrument, timeframe)

    if not candles:
        # A backward page that came back empty is NOT an error: it is the honest
        # "you have reached the start of available history" answer. Return 200 with
        # no candles + has_more_history=False so the chart can say so, rather than a
        # 404 the front would read as "the whole feed broke".
        if before is not None:
            return CandlesResponse(
                instrument=instrument,
                timeframe=timeframe,
                candles=[],
                has_more_history=False,
            )
        # Valid combo, but the cache has no candles yet (combo not bootstrapped /
        # scheduler hasn't filled it). The front shows "graphique indisponible".
        raise HTTPException(
            status_code=404,
            detail=f"No candles cached yet for {instrument}/{timeframe}",
        )

    # Does an older page still exist? True when the oldest candle returned here is
    # NOT the oldest the cache holds for this combo. One cheap indexed MIN(ts) read.
    has_more = False
    try:
        coverage = store.get_coverage(instrument, timeframe)
        oldest_returned = candles[0].ts
        if oldest_returned.tzinfo is None:
            oldest_returned = oldest_returned.replace(tzinfo=timezone.utc)
        if coverage.oldest_ts is not None:
            cov_oldest = coverage.oldest_ts
            if cov_oldest.tzinfo is None:
                cov_oldest = cov_oldest.replace(tzinfo=timezone.utc)
            has_more = oldest_returned > cov_oldest
    except Exception:
        # Coverage is a nicety for the "start of history" notice — never fail the
        # candle read over it. Default to False (front simply won't page further).
        logger.exception("candles coverage read failed for %s/%s", instrument, timeframe)

    return CandlesResponse(
        instrument=instrument,
        timeframe=timeframe,
        candles=[_to_candle_out(c) for c in candles],
        has_more_history=has_more,
    )


def _resolve_assembler(request: Request):
    """The market-reading assembler on app_state (owns the data provider + store)."""
    app_state = getattr(request.app.state, "app_state", None)
    if app_state is None:
        return None
    return getattr(app_state, "market_reading_assembler", None)


def _resolve_candles_store(request: Request):
    """Reuse the same candles store the assembler/scheduler populate."""
    app_state = getattr(request.app.state, "app_state", None)
    if app_state is None:
        return None
    # Preferred: the store the assembler already owns (single shared instance).
    assembler = getattr(app_state, "market_reading_assembler", None)
    store = getattr(assembler, "candles_store", None) if assembler else None
    if store is not None:
        return store
    # Fallback: a store attached directly to app_state, if any.
    return getattr(app_state, "candles_store", None)


def _to_candle_out(candle) -> CandleOut:
    """Serialise a domain Candle to UTC-epoch-seconds JSON.

    candles_cache stores UTC timestamps; if a parsed ts is tz-naive we treat it
    as UTC (never local) so the chart never drifts by the server's offset.
    """
    ts = candle.ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return CandleOut(
        time=int(ts.timestamp()),
        open=float(candle.open),
        high=float(candle.high),
        low=float(candle.low),
        close=float(candle.close),
        volume=float(candle.volume) if candle.volume is not None else 0.0,
    )
