"""Calendar service (NW-1 / NW-1b) — provider-agnostic orchestration.

Pipeline: a ``CalendarProvider`` yields neutral ``ProviderEvent``s → the service
attaches followed markets (config/event_market_map.json, the single documented
rule) → persists them (``CalendarCacheStore``) → serves a chronological window
plus honest coverage, the per-source freshness, and the licence-required
attribution block. The service depends ONLY on the provider interface: it never
reads a provider-specific field, so swapping/adding an adapter changes nothing
here.

Market attachment (the documented rule): an event is attached to every market
whose ``driver_currencies`` include the event's currency. An event attached to
NO followed market is dropped — never attached by default, never displayed.

Resilience (mission NW-1b §2A): a provider failure keeps the cache; a source
that returns nothing is not erased — it is reported as not-refreshed with the
date of its last successful refresh.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from src.intelligence.calendar_providers import CalendarProvider, build_calendar_provider
from src.intelligence.calendar_providers.base import ProviderEvent
from src.intelligence.calendar_providers.values import (
    MultiValueFetcher,
    build_value_fetcher,
)
from src.intelligence.calendar_schema import (
    CalendarAttribution,
    CalendarCoverage,
    CalendarEvent,
    CalendarResponse,
    CalendarSeriesPoint,
    compute_value_state,
)
from src.storage.calendar_cache_store import CalendarCacheEvent, CalendarCacheStore

logger = logging.getLogger(__name__)

_ENV_MAP_PATH = "CALENDAR_EVENT_MAP_PATH"

# Sentinel so `value_fetcher=None` (explicitly disable) is distinct from the
# default (build the env-configured fetcher).
_UNSET = object()

# Published-history (curve) cache TTL. A monthly series changes at most monthly,
# so 6h keeps the live BLS calls to a handful per day — well under the free
# 500/day key ceiling — while still surfacing a fresh point the same day.
SERIES_CACHE_TTL_SECONDS = int(os.environ.get("CALENDAR_SERIES_CACHE_TTL_S", "21600"))


def _default_map_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "event_market_map.json"


def load_market_map(path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Load the documented market → driver-currencies rule as an ordered map.

    Returns ``{market: [currency, ...]}`` preserving config order. On any error
    the map is empty (⇒ nothing is attached ⇒ nothing displayed), which is the
    honest failure mode: never invent an attachment.
    """
    p = path or (
        Path(os.environ[_ENV_MAP_PATH]) if os.environ.get(_ENV_MAP_PATH) else _default_map_path()
    )
    try:
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        markets = raw.get("markets", {})
        out: Dict[str, List[str]] = {}
        for market, spec in markets.items():
            drivers = spec.get("driver_currencies", []) if isinstance(spec, dict) else []
            out[str(market)] = [str(c).upper() for c in drivers]
        return out
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.error("failed to load event_market_map (%s): %s — empty map", p, exc)
        return {}


class CalendarService:
    """Fetches (via a provider), attaches markets, caches, and serves the
    calendar window + coverage + attribution."""

    DEFAULT_TTL_SECONDS = 120
    DEFAULT_LOOKAHEAD_MIN = 7 * 24 * 60   # 7 days forward
    DEFAULT_LOOKBACK_MIN = 3 * 24 * 60    # 3 days back
    # A source silent for longer than this is flagged "not refreshed recently".
    # 24h: the .ics/data feeds are polled on the TTL cadence, so a full day of
    # silence is a genuine retrieval delay worth surfacing.
    STALE_AFTER_HOURS = 24
    # CAL-1 root fix — the refresh does a slow network fan-out (up to 6 .ics
    # fetches + one value-API call per event, 10-12s each). It must NEVER run in
    # the request thread when we already have a cache to serve: a stale cache is
    # refreshed in the BACKGROUND and served immediately. Only a genuinely COLD
    # cache (nothing to serve) waits for the first fill — and even then, only for
    # this bounded budget, so a request can never exceed the client's timeout.
    COLD_FILL_BUDGET_S = 3.0

    def __init__(
        self,
        provider: Optional[CalendarProvider] = None,
        store: Optional[CalendarCacheStore] = None,
        market_map: Optional[Dict[str, List[str]]] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        value_fetcher: Optional["MultiValueFetcher"] = _UNSET,  # type: ignore[assignment]
    ) -> None:
        self._provider = provider if provider is not None else build_calendar_provider()
        self._store = store if store is not None else CalendarCacheStore()
        self._market_map = market_map if market_map is not None else load_market_map()
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        # None disables enrichment; _UNSET (default) builds the env-configured one.
        self._value_fetcher = (
            build_value_fetcher() if value_fetcher is _UNSET else value_fetcher
        )
        self._coverage: Tuple[Optional[datetime], Optional[datetime]] = (None, None)
        # Single-flight guard for the background refresh (CAL-1): at most one
        # network refresh runs at a time; concurrent requests serve the cache.
        self._refresh_lock = threading.Lock()
        # Per-series published-history cache (NW-7): the twelve-figure curve is a
        # monthly series, so one live fetch per series every few hours is ample —
        # WITHOUT it, every detail-page view would spend one BLS request and the
        # free 500/day key would be exhausted. Keyed by (source, series_code, kind).
        self._series_cache: Dict[Tuple[str, str, str], Tuple[datetime, List[CalendarSeriesPoint]]] = {}
        self._series_cache_ttl = timedelta(seconds=SERIES_CACHE_TTL_SECONDS)
        # Latest-value (enrichment) cache (NW-7b): _enrich_values ran uncached on
        # EVERY refresh (TTL 120s) × every key-gated series, which alone exhausted
        # the BLS free 500/day key — so the curve's own fetch was then always
        # rejected. Cache the published latest value per (source, series_code) for
        # the same window; the None (unreachable / quota) result is cached too, so
        # a spent quota is not re-hammered every two minutes.
        self._value_point_cache: Dict[Tuple[str, str], Tuple[datetime, object]] = {}

    # ------------------------------------------------------------------ #
    # Market attachment (the documented rule)
    # ------------------------------------------------------------------ #
    def attach_markets(self, currency: str) -> List[str]:
        """Markets whose driver-currencies include ``currency`` (config order)."""
        cur = (currency or "").upper()
        return [m for m, drivers in self._market_map.items() if cur in drivers]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_calendar(
        self,
        now: Optional[datetime] = None,
        lookahead_minutes: int = DEFAULT_LOOKAHEAD_MIN,
        lookback_minutes: int = DEFAULT_LOOKBACK_MIN,
    ) -> CalendarResponse:
        now = self._coerce_now(now)
        self._maybe_refresh(now)

        window_start = now - timedelta(minutes=lookback_minutes)
        window_end = now + timedelta(minutes=lookahead_minutes)

        cached = self._store.get_events_between(window_start, window_end)
        events = []
        for e in cached:
            ev = self._to_schema(e)
            # Classify the value into one of the four explicit states (never a
            # bare null) using this request's clock.
            ev.actual_state = compute_value_state(
                ev.series_code, ev.actual, ev.scheduled_at, now
            )
            events.append(ev)

        feed_start, feed_end = self._coverage
        if feed_start is None and feed_end is None:
            feed_start, feed_end = self._store.coverage_bounds()
        # Forward-looking only: "partial" means the window reaches BEYOND the last
        # known release. A window that starts before the earliest event is not
        # "uncovered" (there simply are no past events) — it must not raise the
        # coverage banner on a forward calendar.
        partial = bool(feed_end is not None and window_end > feed_end)

        last_success, stale = self._freshness(now)
        coverage = CalendarCoverage(
            source=self._provider.source_name,
            feed_start=feed_start,
            feed_end=feed_end,
            partial=partial,
            last_success=last_success,
            stale_sources=stale,
        )
        return CalendarResponse(
            events=events,
            window_start=window_start,
            window_end=window_end,
            coverage=coverage,
            attribution=self._attribution_for(events),
            generated_at=now,
        )

    def get_calendar_range(
        self,
        window_start: datetime,
        window_end: datetime,
        now: Optional[datetime] = None,
    ) -> CalendarResponse:
        """Serve every attached event in an ARBITRARY [start, end] range — the
        month grid needs whole months forward AND backward, which the now-relative
        lookahead/lookback window cannot express. Same shape, coverage and
        attribution as ``get_calendar``; values are classified with this request's
        clock. No per-event series is attached (that is the detail path only)."""
        now = self._coerce_now(now)
        window_start = self._coerce_now(window_start)
        window_end = self._coerce_now(window_end)
        self._maybe_refresh(now)

        cached = self._store.get_events_between(window_start, window_end)
        events = []
        for e in cached:
            ev = self._to_schema(e)
            ev.actual_state = compute_value_state(
                ev.series_code, ev.actual, ev.scheduled_at, now
            )
            events.append(ev)

        feed_start, feed_end = self._coverage
        if feed_start is None and feed_end is None:
            feed_start, feed_end = self._store.coverage_bounds()
        partial = bool(feed_end is not None and window_end > feed_end)

        last_success, stale = self._freshness(now)
        coverage = CalendarCoverage(
            source=self._provider.source_name,
            feed_start=feed_start,
            feed_end=feed_end,
            partial=partial,
            last_success=last_success,
            stale_sources=stale,
        )
        return CalendarResponse(
            events=events,
            window_start=window_start,
            window_end=window_end,
            coverage=coverage,
            attribution=self._attribution_for(events),
            generated_at=now,
        )

    def get_event(
        self, event_id: str, now: Optional[datetime] = None
    ) -> CalendarResponse:
        """Serve ONE event by its stable id, WITHOUT any window (REC point 1).

        The per-event detail page must never depend on a list window: a
        deep-linked event exists by definition. Returns a CalendarResponse whose
        ``events`` holds exactly the matching event (or is empty when the id
        genuinely does not exist), plus its source attribution and freshness —
        the same shape the detail page already renders."""
        now = self._coerce_now(now)
        self._maybe_refresh(now)

        cached = self._store.get_event_by_id(event_id)
        events = []
        if cached is not None:
            ev = self._to_schema(cached)
            ev.actual_state = compute_value_state(
                ev.series_code, ev.actual, ev.scheduled_at, now
            )
            # The published history (twelve-figure curve) is attached ONLY here, on
            # the per-event detail path — one series call per detail, never per row
            # of the list/month window. Absent series/fetcher ⇒ empty ⇒ no curve.
            if self._value_fetcher is not None and ev.series_code:
                from src.intelligence.calendar_providers.official_sources.base_official import (
                    series_kind_for,
                )

                kind = series_kind_for(ev.series_code)
                cache_key = (ev.source, ev.series_code, kind)
                cached_series = self._series_cache.get(cache_key)
                if (
                    cached_series is not None
                    and now - cached_series[0] < self._series_cache_ttl
                ):
                    ev.value_series = cached_series[1]
                else:
                    series = [
                        CalendarSeriesPoint(period=p.period, value=p.value)
                        for p in self._value_fetcher.series_for(
                            ev.source, ev.series_code, kind=kind
                        )
                    ]
                    # Only cache a NON-EMPTY series: an empty read (transient
                    # network failure or exhausted quota) must not be pinned for
                    # hours — retry on the next view instead.
                    if series:
                        self._series_cache[cache_key] = (now, series)
                    ev.value_series = series
            events.append(ev)

        last_success, stale = self._freshness(now)
        coverage = CalendarCoverage(
            source=self._provider.source_name,
            feed_start=None,
            feed_end=None,
            partial=False,
            last_success=last_success,
            stale_sources=stale,
        )
        return CalendarResponse(
            events=events,
            window_start=now,
            window_end=now,
            coverage=coverage,
            attribution=self._attribution_for(events),
            generated_at=now,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _coerce_now(self, now: Optional[datetime]) -> datetime:
        ts = now if now is not None else self._clock()
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def _freshness(self, now: datetime) -> Tuple[Dict[str, datetime], List[str]]:
        """Per-source last success + the sources whose data is not fresh. A source
        is stale if it was NOT refreshed in the latest cycle OR its last success is
        older than ``STALE_AFTER_HOURS`` — a retrieval delay is never silent. Rows
        are kept regardless; only the freshness flag changes."""
        last_success = self._store.source_last_success()
        if not last_success:
            return ({}, [])
        newest = max(last_success.values())
        cutoff = now - timedelta(hours=self.STALE_AFTER_HOURS)
        stale = sorted(
            s for s, ts in last_success.items() if ts < newest or ts < cutoff
        )
        return (last_success, stale)

    def _attribution_for(self, events: List[CalendarEvent]) -> List[CalendarAttribution]:
        """One attribution entry per source that actually produced a served
        event — a licence condition, asserted in tests. A served event whose
        source has no attribution is a bug the test catches."""
        present = {e.source for e in events}
        out: List[CalendarAttribution] = []
        seen = set()
        for att in self._provider.attributions():
            if att.source in present and att.source not in seen:
                seen.add(att.source)
                out.append(
                    CalendarAttribution(
                        source=att.source,
                        organism=att.organism,
                        license_label=att.license_label,
                        policy_url=att.policy_url,
                    )
                )
        return out

    def _maybe_refresh(self, now: datetime) -> None:
        """Refresh the cache if the TTL has elapsed — WITHOUT blocking the request
        on the network (CAL-1 root fix).

        The refresh (up to 6 ``.ics`` fetches + a value-API call per event, 10-12s
        each) runs single-flight in a BACKGROUND thread; the request serves the
        cache we already have. Only a genuinely COLD cache (nothing to serve) waits
        for the first fill, and only for ``COLD_FILL_BUDGET_S`` — so no request can
        ever exceed the client's timeout, and a slow official feed can never leave
        the calendar stuck loading."""
        last = self._store.last_fetch_at()
        if last is not None and (now - last).total_seconds() < self._ttl_seconds:
            return
        thread = self._start_background_refresh(now)
        if thread is not None and last is None:
            # Cold cache: wait a bounded budget so the very first load is not
            # empty, but never longer — the rest completes in the background.
            thread.join(self.COLD_FILL_BUDGET_S)

    def _start_background_refresh(self, now: datetime) -> Optional[threading.Thread]:
        """Kick a single-flight refresh in a daemon thread. Returns the started
        thread, or None when a refresh is already in flight (its work is shared)."""
        if not self._refresh_lock.acquire(blocking=False):
            return None

        def _run() -> None:
            try:
                self._do_refresh(now)
            except Exception as exc:  # a background failure must never escape
                logger.warning("calendar background refresh failed: %s", exc)
            finally:
                self._refresh_lock.release()

        try:
            thread = threading.Thread(
                target=_run, name="calendar-refresh", daemon=True
            )
            thread.start()
        except Exception:
            self._refresh_lock.release()
            raise
        return thread

    def _do_refresh(self, now: datetime) -> None:
        """Fetch from the provider, attach markets, persist. Runs off the request
        thread. Persists the DATES first (so the grid can render as soon as the
        feed answers), then enriches published VALUES (the slower per-series pass)
        and persists again. A provider failure keeps the existing cache."""
        try:
            fetch = self._provider.fetch()
        except Exception as exc:  # defensive — provider failure keeps stale cache
            logger.warning("calendar provider fetch failed: %s — keeping cache", exc)
            return

        self._coverage = (fetch.coverage_start, fetch.coverage_end)

        persisted: List[CalendarCacheEvent] = []
        for ev in fetch.events:
            markets = self.attach_markets(ev.currency)
            if not markets:
                continue  # no followed market → never attached, never displayed
            persisted.append(self._to_cache(ev, markets))
        if not persisted:
            return
        # Dates first — the month grid needs only the schedule, not the values.
        self._store.upsert_events(persisted, fetched_at=now)
        # Then the slower published-value pass; re-persist with values attached.
        self._enrich_values(persisted, now)
        self._store.upsert_events(persisted, fetched_at=now)

    def _enrich_values(self, events: List[CalendarCacheEvent], now: datetime) -> None:
        """Fetch published values by stable series code — never by title, never
        fabricated. Two cases:

        · PAST release (already published): fill its own ``actual`` (the value
          published for that release) and ``previous`` (the prior period).
        · FUTURE release (not published yet → stays ``pending``): its own value
          does not exist, but the PREVIOUS period's value IS published. The latest
          published observation (``point.actual``) is exactly that prior period, so
          it fills ``previous`` — the upcoming release shows the last known figure
          for context while its own value remains ``pending``.

        A series-less event is left ``unavailable``; an unreachable fetch leaves
        the event ``unfetched`` (its ``fetched_at`` records the attempt). The
        store's upsert flags a revision only if a stored ``actual`` later changes."""
        fetcher = self._value_fetcher
        if fetcher is None:
            return
        _MISS = self._value_point_cache  # (source, series_code) → (at, point|None)
        for i, ce in enumerate(events):
            if not ce.series_code or ce.actual is not None:
                continue
            key = (ce.source, ce.series_code)
            cached = _MISS.get(key)
            if cached is not None and now - cached[0] < self._series_cache_ttl:
                point = cached[1]  # may be None (a recent unreachable/quota result)
            else:
                point = fetcher.fetch_for(ce.source, ce.series_code)
                # Cache success AND failure for the window: an exhausted BLS quota
                # must not be re-hit on every 120s refresh (that kept it exhausted).
                _MISS[key] = (now, point)
            if point is None:
                continue  # stays unfetched — the attempt is recorded via fetched_at
            if ce.scheduled_at > now:
                # Upcoming: leave ``actual`` pending; the latest published value is
                # the period BEFORE this release → it is the "previous".
                if point.actual is not None:
                    events[i] = dataclasses.replace(ce, previous=point.actual)
            else:
                events[i] = dataclasses.replace(
                    ce,
                    actual=point.actual,
                    previous=point.previous if point.previous is not None else ce.previous,
                )

    def _to_cache(self, ev: ProviderEvent, markets: List[str]) -> CalendarCacheEvent:
        return CalendarCacheEvent(
            event_id=f"{ev.source}:{ev.provider_ref}",
            source=ev.source,
            event=ev.event,
            currency=ev.currency,
            scheduled_at=ev.scheduled_at,
            markets=markets,
            series_code=ev.series_code,
            license_label=ev.license_label,
            organism=ev.organism,
            periodicity=ev.periodicity,
            source_timezone=ev.source_timezone,
            time_confirmed=ev.time_confirmed,
            value_unit=ev.value_unit,
            actual=ev.actual,
            actual_initial=ev.actual_initial,
            previous=ev.previous,
            revised=ev.revised,
            revised_at=ev.revised_at,
        )

    @staticmethod
    def _to_schema(e: CalendarCacheEvent) -> CalendarEvent:
        return CalendarEvent(
            event_id=e.event_id,
            source=e.source,
            series_code=e.series_code,
            license_label=e.license_label,
            event=e.event,
            currency=e.currency,
            organism=e.organism,
            periodicity=e.periodicity,  # type: ignore[arg-type]
            scheduled_at=e.scheduled_at,
            source_timezone=e.source_timezone,
            time_confirmed=e.time_confirmed,
            markets=e.markets,
            value_unit=e.value_unit,
            actual=e.actual,
            actual_initial=e.actual_initial,
            previous=e.previous,
            revised=e.revised,
            revised_at=e.revised_at,
            refreshed_at=e.refreshed_at,
            # actual_state is computed in get_calendar (needs the request clock).
        )


__all__ = ["CalendarService", "load_market_map"]
