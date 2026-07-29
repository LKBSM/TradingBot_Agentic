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

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from src.intelligence.calendar_providers import CalendarProvider, build_calendar_provider
from src.intelligence.calendar_providers.base import ProviderEvent
from src.intelligence.calendar_schema import (
    CalendarAttribution,
    CalendarCoverage,
    CalendarEvent,
    CalendarResponse,
)
from src.storage.calendar_cache_store import CalendarCacheEvent, CalendarCacheStore

logger = logging.getLogger(__name__)

_ENV_MAP_PATH = "CALENDAR_EVENT_MAP_PATH"


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

    def __init__(
        self,
        provider: Optional[CalendarProvider] = None,
        store: Optional[CalendarCacheStore] = None,
        market_map: Optional[Dict[str, List[str]]] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._provider = provider if provider is not None else build_calendar_provider()
        self._store = store if store is not None else CalendarCacheStore()
        self._market_map = market_map if market_map is not None else load_market_map()
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._coverage: Tuple[Optional[datetime], Optional[datetime]] = (None, None)

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
        events = [self._to_schema(e) for e in cached]

        feed_start, feed_end = self._coverage
        if feed_start is None and feed_end is None:
            feed_start, feed_end = self._store.coverage_bounds()
        # Forward-looking only: "partial" means the window reaches BEYOND the last
        # known release. A window that starts before the earliest event is not
        # "uncovered" (there simply are no past events) — it must not raise the
        # coverage banner on a forward calendar.
        partial = bool(feed_end is not None and window_end > feed_end)

        last_success, stale = self._freshness()
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

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _coerce_now(self, now: Optional[datetime]) -> datetime:
        ts = now if now is not None else self._clock()
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def _freshness(self) -> Tuple[Dict[str, datetime], List[str]]:
        """Per-source last success + the sources that were NOT refreshed in the
        latest successful cycle (their rows are kept, they are flagged stale)."""
        last_success = self._store.source_last_success()
        if not last_success:
            return ({}, [])
        newest = max(last_success.values())
        stale = sorted(s for s, ts in last_success.items() if ts < newest)
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
        last = self._store.last_fetch_at()
        if last is not None and (now - last).total_seconds() < self._ttl_seconds:
            return
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
        if persisted:
            self._store.upsert_events(persisted, fetched_at=now)

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
        )


__all__ = ["CalendarService", "load_market_map"]
