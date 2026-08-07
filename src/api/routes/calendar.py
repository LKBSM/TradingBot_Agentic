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
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException, Path, Query, Request

from src.intelligence.calendar_schema import CalendarResponse
from src.intelligence.publication_measures import MEASURED_MARKETS as _MEASURABLE_MARKETS
from src.intelligence.publication_measures_schema import PublicationMeasures

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["calendar"])

_MAX_LOOKAHEAD_DAYS = 30
_MAX_LOOKBACK_DAYS = 30

_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")

# ``_MEASURABLE_MARKETS`` is the single source in publication_measures.py
# (MEASURED_MARKETS) — imported above so the API gate, the measures computation
# and the automatic deep-history maintainer all agree. An event_key absent there
# has no measures and the page renders none (never a placeholder).

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


def prewarm_publication_measures() -> None:
    """Compute EVERY measured publication's measures once and fill the cache, so
    the first interactive request is instant. The compute is a multi-second engine
    replay per publication (which otherwise overruns the client's fetch timeout on
    the very first page view). Safe to run in a background thread at boot; runs
    only after the deep-history seed has populated candles.db. Never raises."""
    from src.intelligence.publication_measures import load_default_measures

    for event_key, market in list(_MEASURABLE_MARKETS.items()):
        try:
            measures = load_default_measures(event_key=event_key, market=market)
            _MEASURES_CACHE[event_key] = (datetime.now(timezone.utc), measures)
            logger.info(
                "prewarmed measures %s: has_any=%s",
                event_key, bool(measures and measures.has_any()),
            )
        except Exception:  # one publication must not abort the warm
            logger.exception("prewarm measures failed for %s", event_key)


@router.get("/publications/{event_key}/measures/debug")
def get_publication_measures_debug(
    event_key: str = Path(..., description="Recurring event key, e.g. 'us_cpi'"),
) -> Dict[str, Any]:
    """Read-only triage (NW-7c): WHY the measures do/don't compute in this env —
    which release/candle source yields data and whether they overlap. No secrets,
    no side effects. Returns a plain JSON object; never raises."""
    from src.intelligence.publication_measures import diagnose_default_measures

    market = _MEASURABLE_MARKETS.get(event_key) or "XAUUSD"
    return diagnose_default_measures(event_key=event_key, market=market)


# One-time DEEP backfill of the measured market's M15 history into candles.db, so
# the measures have the months of intraday history they need (NW-7d). The bundled
# CSV is absent in prod and Twelve Data caps a single request at ~52 days, so we
# page backward. Runs in the background (paginated + rate-limited ≈ 1–2 min);
# single-flight; clears the measures cache on completion so they recompute.
_BACKFILL_STATE: Dict[str, Any] = {"running": False, "last_result": None}


@router.get("/publications/{event_key}/measures/backfill")
def trigger_measures_backfill(
    event_key: str = Path(..., description="Recurring event key, e.g. 'us_cpi'"),
    months: int = Query(14, ge=3, le=36, description="Months of M15 history to fill"),
) -> Dict[str, Any]:
    market = _MEASURABLE_MARKETS.get(event_key)
    if market is None:
        raise HTTPException(status_code=422, detail="event not measurable")
    if _BACKFILL_STATE["running"]:
        return {"status": "already_running", "last_result": _BACKFILL_STATE["last_result"]}

    target_bars = int(months) * 2200  # ~M15 bars per trading month (5d/wk)

    def _run() -> None:
        _BACKFILL_STATE["running"] = True
        try:
            from src.intelligence.data_providers import TwelveDataProvider
            from src.intelligence.history_backfill import deep_backfill_combo
            from src.storage.candles_cache_store import CandlesCacheStore

            provider = TwelveDataProvider()  # reads TWELVE_DATA_API_KEY from env
            store = CandlesCacheStore()      # reads CANDLES_DB_PATH (mounted disk)
            res = deep_backfill_combo(
                provider, store, market, "M15", target_bars=target_bars
            )
            _BACKFILL_STATE["last_result"] = res
            _MEASURES_CACHE.pop(event_key, None)  # drop cached None → recompute
        except Exception as exc:  # never crash the worker thread
            logger.exception("measures backfill failed for %s", event_key)
            _BACKFILL_STATE["last_result"] = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            _BACKFILL_STATE["running"] = False

    threading.Thread(target=_run, name=f"backfill-{market}-M15", daemon=True).start()
    return {
        "status": "started", "market": market, "timeframe": "M15",
        "months": months, "target_bars": target_bars,
        "note": "Check .../measures/backfill/status for progress/errors.",
    }


@router.get("/publications/{event_key}/measures/backfill/status")
def measures_backfill_status(
    event_key: str = Path(..., description="Recurring event key, e.g. 'us_cpi'"),
) -> Dict[str, Any]:
    """Progress/last result of the deep backfill WITHOUT starting a new run — so a
    failure (auth, provider error, depth) is visible instead of silent."""
    return {
        "running": bool(_BACKFILL_STATE["running"]),
        "last_result": _BACKFILL_STATE["last_result"],
    }


# One-time IMPORT of a deep price-history CSV from a URL onto the mounted disk, so
# the measures have months of XAU M15 WITHOUT spending Twelve Data credits (the
# free 800/day quota is over-subscribed). Streams to a temp file, validates it is
# an OHLC CSV, then atomically swaps it into place under the configured name so
# ``_find_gold_csv`` picks it up. Background + single-flight + status.
_IMPORT_STATE: Dict[str, Any] = {"running": False, "last_result": None}
_IMPORT_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB safety cap


def _measures_data_dir() -> str:
    """The mounted data directory where candles.db lives (so the CSV lands where
    ``_find_gold_csv`` looks), env-resolved — NOT config's hard-coded path."""
    import os

    cdb = os.environ.get("CANDLES_DB_PATH")
    if cdb and os.path.dirname(cdb):
        return os.path.dirname(cdb)
    if os.environ.get("DATA_DIR"):
        return os.environ["DATA_DIR"]
    try:
        import config as _config
        return getattr(_config, "DATA_DIR", ".")
    except Exception:  # pragma: no cover
        return "."


@router.get("/publications/{event_key}/measures/import-csv")
def import_measures_csv(
    event_key: str = Path(..., description="Recurring event key, e.g. 'us_cpi'"),
    url: str = Query(..., description="Direct download URL of the XAU M15 OHLC CSV"),
) -> Dict[str, Any]:
    if _MEASURABLE_MARKETS.get(event_key) is None:
        raise HTTPException(status_code=422, detail="event not measurable")
    if _IMPORT_STATE["running"]:
        return {"status": "already_running", "last_result": _IMPORT_STATE["last_result"]}

    def _run() -> None:
        import os

        _IMPORT_STATE["running"] = True
        try:
            import requests

            import config as _config

            data_dir = _measures_data_dir()
            os.makedirs(data_dir, exist_ok=True)
            name = os.path.basename(
                getattr(_config, "HISTORICAL_DATA_FILE", "") or "XAU_15MIN_2019_2026.csv"
            )
            target = os.path.join(data_dir, name)
            tmp = target + ".part"
            total = 0
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > _IMPORT_MAX_BYTES:
                            raise ValueError("download exceeds 2 GB cap")
                        fh.write(chunk)
            # Validate it is an OHLC CSV BEFORE swapping it into place.
            with open(tmp, "r", encoding="utf-8", errors="replace") as fh:
                header = fh.readline().strip().lower()
            cols = {c.strip().strip('"') for c in re.split(r"[,;\t]", header)}
            has_time = bool(cols & {"date", "datetime", "time", "timestamp"})
            has_ohlc = {"open", "high", "low", "close"} <= cols
            if not (has_time and has_ohlc):
                os.remove(tmp)
                raise ValueError(f"not an OHLC CSV (header: {header[:120]})")
            os.replace(tmp, target)  # atomic — readers never see a partial file
            _MEASURES_CACHE.pop(event_key, None)  # drop cached None → recompute
            _IMPORT_STATE["last_result"] = {
                "ok": True, "path": target, "bytes": total, "header": header[:120],
            }
        except Exception as exc:
            logger.exception("measures CSV import failed for %s", event_key)
            _IMPORT_STATE["last_result"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        finally:
            _IMPORT_STATE["running"] = False

    threading.Thread(target=_run, name="import-csv", daemon=True).start()
    return {
        "status": "started",
        "note": "Check .../measures/import-csv/status; then reload the IPC page.",
    }


@router.get("/publications/{event_key}/measures/import-csv/status")
def import_measures_csv_status(
    event_key: str = Path(..., description="Recurring event key, e.g. 'us_cpi'"),
) -> Dict[str, Any]:
    return {
        "running": bool(_IMPORT_STATE["running"]),
        "last_result": _IMPORT_STATE["last_result"],
    }


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
