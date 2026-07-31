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

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query, Request

from src.intelligence.calendar_schema import CalendarResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["calendar"])

_MAX_LOOKAHEAD_DAYS = 30
_MAX_LOOKBACK_DAYS = 30


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
