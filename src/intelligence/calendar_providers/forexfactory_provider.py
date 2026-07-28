"""ForexFactory calendar adapter — PROTOTYPE / LOCAL DEVELOPMENT ONLY.

⚠️  NOT A PRODUCT SOURCE. ForexFactory has NO official API and its terms grant
    NO COMMERCIAL DISPLAY RIGHTS; the feed also lacks a stable recurring event
    id, the issuing organism, the value unit and revisions. It is wired here
    purely so the calendar UI can be exercised locally against real-shaped data
    while the official-source integration (BLS/BEA/Census/Fed, Eurostat/ECB) is
    built. It is NEVER the default source (see the provider factory) and must
    not be enabled in any client-facing deployment.

All ForexFactory-specific parsing stays inside this file: nothing FF-shaped
crosses the CalendarProvider boundary. The service, store, schema and front see
only neutral ``ProviderEvent``s.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from src.intelligence.calendar_providers.base import (
    CalendarProvider,
    ProviderEvent,
    ProviderFetch,
)

# FF fetch + parse helpers are reused (no network/parse duplication); they are
# the ONLY FF-specific imports and they live behind this adapter.
from src.intelligence.news_pipeline import (
    _default_ff_fetch,
    _event_id,
    _parse_ff_datetime,
    _parse_numeric,
)

logger = logging.getLogger(__name__)

# Calendar keeps the full impact range (the page's "Faible" filter must be
# honest). holiday / non-economic / empty are dropped.
_FF_IMPACT_KEEP = {"high", "medium", "low"}

_LICENSE_LABEL = (
    "ForexFactory (prototype dev) — aucun droit d'affichage commercial"
)
# ForexFactory publishes in US Eastern; the neutral event records that as the
# original publication timezone.
_FF_SOURCE_TZ = "America/New_York"


class ForexFactoryCalendarProvider(CalendarProvider):
    """Prototype adapter over the public ForexFactory JSON feed. Dev-only."""

    def __init__(self, fetch_fn: Optional[Callable[[], List[dict]]] = None) -> None:
        self._fetch_fn = fetch_fn or _default_ff_fetch

    @property
    def source_name(self) -> str:
        return "forexfactory"

    def fetch(self) -> ProviderFetch:
        try:
            raw = self._fetch_fn()
        except Exception as exc:  # defensive — any fetcher failure is graceful
            logger.warning("ForexFactory fetch failed: %s — empty fetch", exc)
            return ProviderFetch(events=[])

        events: List[ProviderEvent] = []
        for item in raw or []:
            ev = self._normalize(item)
            if ev is not None:
                events.append(ev)

        if not events:
            return ProviderFetch(events=[])
        starts = [e.scheduled_at for e in events]
        return ProviderFetch(
            events=events,
            coverage_start=min(starts),
            coverage_end=max(starts),
        )

    def _normalize(self, ev: dict) -> Optional[ProviderEvent]:
        if not isinstance(ev, dict):
            return None
        title = (ev.get("title") or "").strip()
        currency = (ev.get("country") or "").strip().upper()
        date_str = ev.get("date") or ""
        if not (title and currency and date_str):
            return None

        impact = (ev.get("impact") or "").lower()
        if impact not in _FF_IMPACT_KEEP:
            return None

        scheduled_at = _parse_ff_datetime(date_str)
        if scheduled_at is None:
            return None

        return ProviderEvent(
            source=self.source_name,
            provider_ref=_event_id(title, currency, scheduled_at),
            series_code=None,          # FF has no stable recurring id
            event=title,
            currency=currency,
            impact=impact,
            scheduled_at=scheduled_at,
            source_timezone=_FF_SOURCE_TZ,
            organism=None,             # FF does not name the issuing organism
            value_unit=None,           # FF does not carry a clean unit
            actual=_parse_numeric(ev.get("actual")),
            forecast=_parse_numeric(ev.get("forecast")),
            previous=_parse_numeric(ev.get("previous")),
            revised=False,
            previous_before_revision=None,
            license_label=_LICENSE_LABEL,
        )


__all__ = ["ForexFactoryCalendarProvider"]
