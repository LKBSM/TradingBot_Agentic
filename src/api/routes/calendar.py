"""Economic-calendar endpoints (NW-1 — "Actualités programmées").

GET /api/calendar?lookahead_days=7&lookback_days=3
  → 200 + CalendarResponse (chronological events attached to a followed market,
    plus honest coverage).
GET /api/calendar/event/{event_id}
  → 200 + CalendarResponse holding exactly the requested event (or empty when the
    id genuinely does not exist), independent of any window (REC point 1).

Descriptive only. The endpoint announces scheduled MOMENTS and, per record, the
source + organism when known — never a predicted value, direction or reaction.
The active source is env-driven (``CALENDAR_SOURCE``); it defaults to the
official stub (0 events) until the official-source integration lands.
"""

from __future__ import annotations

import calendar as _calendar
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException, Path, Query, Request

from src.intelligence.calendar_schema import CalendarResponse
from src.intelligence.publication_measures_schema import PublicationMeasures

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["calendar"])

_MAX_LOOKAHEAD_DAYS = 30
_MAX_LOOKBACK_DAYS = 30

_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")

# Recurring publications for which engine measures are computed, and the SINGLE
# market each is measured on (the market whose price history we actually hold).
# US CPI is measured on gold; an event_key absent here has no measures and the
# page renders none (never a placeholder). See publication_measures.py.
_MEASURABLE_MARKETS: Dict[str, str] = {"us_cpi": "XAUUSD"}

# Process-local cache of the heavy measures computation (engine replay over past
# releases + CSV reads). Keyed by event_key → (computed_at, measures). Refreshed
# lazily past the TTL; the underlying data is static historical price/calendar.
_MEASURES_CACHE: Dict[str, Tuple[datetime, Optional[PublicationMeasures]]] = {}
_MEASURES_TTL_SECONDS = 3600


def _get_calendar_service(request: Request) -> Any:
    """Return the shared CalendarService, building it lazily on first use.

    Tests inject their own by setting ``app_state.calendar_service`` before the
    call; production builds the env-configured provider (official stub default).
    """
    app_state = getattr(request.app.state, "app_state", None)
    if app_state is None:
        raise HTTPException(status_code=503, detail="Calendar service not configured")
    service = getattr(app_state, "calendar_service", None)
    if service is None:
        from src.intelligence.calendar_service import CalendarService

        service = CalendarService()
        app_state.calendar_service = service
    return service


@router.get("/calendar", response_model=CalendarResponse)
async def get_calendar(
    request: Request,
    lookahead_days: int = Query(7, ge=1, le=_MAX_LOOKAHEAD_DAYS),
    lookback_days: int = Query(3, ge=0, le=_MAX_LOOKBACK_DAYS),
) -> CalendarResponse:
    service = _get_calendar_service(request)
    try:
        return service.get_calendar(
            lookahead_minutes=lookahead_days * 24 * 60,
            lookback_minutes=lookback_days * 24 * 60,
        )
    except Exception:
        logger.exception("calendar generation failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/calendar/month", response_model=CalendarResponse)
def get_calendar_month(
    request: Request,
    month: str = Query(..., description="Target month, 'YYYY-MM'"),
) -> CalendarResponse:
    """Every attached event within one calendar month — the month grid needs whole
    months forward AND backward, which the now-relative window cannot express.
    Sync `def` so the (possibly refreshing) call is threadpooled (REC-1)."""
    m = _MONTH_RE.match(month or "")
    if not m:
        raise HTTPException(status_code=422, detail="month must be 'YYYY-MM'")
    year, mon = int(m.group(1)), int(m.group(2))
    if not (1 <= mon <= 12) or not (1970 <= year <= 2100):
        raise HTTPException(status_code=422, detail="month out of range")
    last_day = _calendar.monthrange(year, mon)[1]
    start = datetime(year, mon, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(year, mon, last_day, 23, 59, 59, tzinfo=timezone.utc)
    service = _get_calendar_service(request)
    try:
        return service.get_calendar_range(start, end)
    except Exception:
        logger.exception("calendar month generation failed for %s", month)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/publications/{event_key}/measures", response_model=PublicationMeasures)
def get_publication_measures(
    event_key: str = Path(..., description="Recurring event key, e.g. 'us_cpi'"),
) -> PublicationMeasures:
    """Engine-measured facts (calm-before / structure-at-T / return-to-calm) for a
    recurring publication. Returns an EMPTY PublicationMeasures (all measures
    None) when the key is not measurable or its data is unavailable — the page
    then renders no measures section, never a placeholder. Cached (heavy compute).
    """
    market = _MEASURABLE_MARKETS.get(event_key)
    if market is None:
        return PublicationMeasures(event_key=event_key, market="")

    now = datetime.now(timezone.utc)
    cached = _MEASURES_CACHE.get(event_key)
    if cached is not None and (now - cached[0]).total_seconds() < _MEASURES_TTL_SECONDS:
        return cached[1] or PublicationMeasures(event_key=event_key, market=market)

    measures: Optional[PublicationMeasures] = None
    try:
        from src.intelligence.publication_measures import load_default_measures

        measures = load_default_measures(event_key=event_key, market=market)
    except Exception:
        logger.exception("measures computation failed for %s", event_key)
        measures = None
    _MEASURES_CACHE[event_key] = (now, measures)
    return measures or PublicationMeasures(event_key=event_key, market=market)


# REC point 1: the per-event detail must load an event by its STABLE ID from
# storage, independent of any list window — a deep-linked event exists by
# definition. Sync `def` so the (possibly refreshing) call is threadpooled and
# never blocks the event loop (REC-1). The response carries the one event (or an
# empty list when the id truly does not exist), so the detail page distinguishes
# "genuinely unknown id" from "outside the current window".
@router.get("/calendar/event/{event_id}", response_model=CalendarResponse)
def get_calendar_event(
    request: Request,
    event_id: str = Path(..., description="Stable event id, e.g. bea:us_gdp:2026-08-26"),
) -> CalendarResponse:
    service = _get_calendar_service(request)
    try:
        return service.get_event(event_id)
    except Exception:
        logger.exception("calendar event fetch failed for %s", event_id)
        raise HTTPException(status_code=500, detail="Internal server error")
