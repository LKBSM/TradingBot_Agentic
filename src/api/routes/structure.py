"""LB-1 — honest coverage + window-bounded structure serving.

Two descriptive, read-only endpoints:

  GET /api/coverage
    → per-combo REAL history in the cache: how many candles, since when, the
      configured target, and whether the fill is complete. Feeds the "historique
      depuis le …" honesty display — a partial period is never shown as complete.

  GET /api/structure?instrument&timeframe&time_from&price_low&price_high
    → only the zones and events that INTERSECT the visible window (+ the totals,
      for the "12 sur 148" counter). The network payload becomes independent of
      how deep storage goes. Backed by the persisted StructureStore; nothing is
      recomputed here.

Neither endpoint calls a provider or runs detection.

DG-1 decision (b) — SINGLE JOURNAL SOURCE (2026-07-29):
  These endpoints are backed by the persisted StructureStore, which today is
  seeded by the backfill only (the scheduler → IncrementalDetector.refresh wiring
  is deferred, cf. AUDIT-lb-1 §4.2), so its journal is FROZEN between backfills.
  The App reads its structure / journal / maturity from the LIVE assembler
  (/api/market-reading) instead — that is the single source of truth on screen.
  These endpoints are therefore intentionally NOT wired to any UI surface. If a
  future surface consumes them, it must first (1) wire the scheduler so the store
  stops being frozen, and (2) reconcile the window (DG-1 point 5) and LABEL the
  covered period — otherwise two journals covering different periods reach the
  same screen (the exact state DG-1 forbids). A guard test locks the "no UI
  consumer" invariant:
  webapp/lib/market-reading/__tests__/single-journal-guard.test.ts.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from src.api.session_auth import optional_account
from src.api.subscription_gate import enforce_access
from src.intelligence import lookback_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["structure"])

# Deep-enough fraction: at/above this share of the target we call a combo's fill
# complete. Below it (or empty) the combo is honestly reported as partial.
_COMPLETE_FRACTION = 0.9


class ComboCoverage(BaseModel):
    instrument: str
    timeframe: str
    candle_count: int
    history_since: Optional[str]   # oldest cached candle (ISO), the honest start
    latest: Optional[str]
    target_bars: int
    duration: str                  # configured depth token, e.g. "6mo"
    complete: bool                 # False ⇒ fill in progress / limited by provider


class CoverageResponse(BaseModel):
    combos: List[ComboCoverage]


class ZoneOut(BaseModel):
    type: str
    direction: str
    level_high: float
    level_low: float
    created_at: Optional[str]
    status: str
    tested: bool
    mitigated_at: Optional[str]
    fill_level: Optional[float]


class EventOut(BaseModel):
    type: str
    direction: str
    level: Optional[float]
    event_ts: str


class StructureWindowResponse(BaseModel):
    instrument: str
    timeframe: str
    zones: List[ZoneOut]
    events: List[EventOut]
    zones_total: int   # surfaced zones for the combo (denominator of "N sur M")
    events_total: int


def _resolve_candles_store(request: Request):
    app_state = getattr(request.app.state, "app_state", None)
    if app_state is None:
        return None
    assembler = getattr(app_state, "market_reading_assembler", None)
    store = getattr(assembler, "candles_store", None) if assembler else None
    return store or getattr(app_state, "candles_store", None)


_default_structure_store = None


def _resolve_structure_store(request: Request):
    app_state = getattr(request.app.state, "app_state", None)
    store = getattr(app_state, "structure_store", None) if app_state else None
    if store is not None:
        return store
    # Lazily fall back to the default-path store (populated by the backfill /
    # incremental driver). Cached so we do not reopen the DB per request.
    global _default_structure_store
    if _default_structure_store is None:
        from src.storage.structure_store import StructureStore
        _default_structure_store = StructureStore()
    return _default_structure_store


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        # Accept epoch seconds or ISO 8601.
        if value.isdigit():
            return datetime.fromtimestamp(int(value), tz=timezone.utc)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


@router.get("/coverage", response_model=CoverageResponse)
async def get_coverage(
    request: Request,
    account: Optional[Dict[str, Any]] = Depends(optional_account),
) -> CoverageResponse:
    # Paid-only gate (no-op while OFF): market-history metadata is market data.
    enforce_access(request, account)
    store = _resolve_candles_store(request)
    combos: List[ComboCoverage] = []
    for instrument, timeframe in lookback_config.enabled_combos():
        depth = lookback_config.depth_for(instrument, timeframe)
        count, oldest, newest = 0, None, None
        if store is not None:
            try:
                cov = store.get_coverage(instrument, timeframe)
                count, oldest, newest = cov.count, cov.oldest_ts, cov.newest_ts
            except Exception:
                logger.exception("coverage read failed for %s/%s", instrument, timeframe)
        combos.append(ComboCoverage(
            instrument=instrument,
            timeframe=timeframe,
            candle_count=count,
            history_since=oldest.isoformat() if oldest else None,
            latest=newest.isoformat() if newest else None,
            target_bars=depth.target_bars,
            duration=depth.duration_str,
            complete=count >= int(depth.target_bars * _COMPLETE_FRACTION),
        ))
    return CoverageResponse(combos=combos)


@router.get("/structure", response_model=StructureWindowResponse)
async def get_structure(
    request: Request,
    instrument: str = Query(...),
    timeframe: str = Query(...),
    time_from: Optional[str] = Query(None, description="ISO or epoch-seconds; window start"),
    price_low: Optional[float] = Query(None),
    price_high: Optional[float] = Query(None),
    account: Optional[Dict[str, Any]] = Depends(optional_account),
) -> StructureWindowResponse:
    # Paid-only gate (no-op while OFF): zones/events are market data.
    enforce_access(request, account)
    if (instrument, timeframe) not in set(lookback_config.enabled_combos()) and \
            instrument not in lookback_config.supported_instruments():
        raise HTTPException(status_code=400, detail=f"Unsupported combo {instrument}/{timeframe}")

    store = _resolve_structure_store(request)
    tf_from = _parse_time(time_from)
    zones = store.get_zones_in_window(
        instrument, timeframe,
        time_from=tf_from, price_low=price_low, price_high=price_high, surfaced_only=True,
    )
    events = store.get_events_in_window(instrument, timeframe, time_from=tf_from)
    return StructureWindowResponse(
        instrument=instrument,
        timeframe=timeframe,
        zones=[_zone_out(z) for z in zones],
        events=[_event_out(e) for e in events],
        zones_total=store.count_zones(instrument, timeframe, surfaced_only=True),
        events_total=len(store.get_event_journal(instrument, timeframe)),
    )


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if isinstance(dt, datetime) else None


def _zone_out(z) -> ZoneOut:
    return ZoneOut(
        type=z.zone_type, direction=z.direction,
        level_high=z.level_high, level_low=z.level_low,
        created_at=_iso(z.created_at), status=z.status, tested=z.tested,
        mitigated_at=_iso(z.mitigated_at), fill_level=z.fill_level,
    )


def _event_out(e: dict) -> EventOut:
    return EventOut(
        type=e["event_type"], direction=e["direction"],
        level=e["level"], event_ts=e["event_ts"],
    )
