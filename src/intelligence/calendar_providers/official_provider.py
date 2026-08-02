"""Official-source calendar provider — the DEFAULT source (NW-1 / NW-1b).

NW-1 shipped this as a stub (0 events). NW-1b turns it into the AGGREGATOR of
the per-organism adapters (BLS, BEA, Census, Federal Reserve, Eurostat, ECB):
the service still sees a single ``CalendarProvider`` named "official", while each
emitted event keeps its own source, organism and licence, and the attribution
block is the union of the organisms that actually produced an event.

Dated releases come from each adapter's ``date_source`` (default: the versioned
``config/calendar_schedule.json``, which ships empty — so absent an entered
schedule or an injected live feed, the calendar is honestly empty rather than
showing fabricated dates). Any single organism failing is swallowed; it never
erases another source's data.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from src.intelligence.calendar_providers.base import (
    CalendarProvider,
    ProviderAttribution,
    ProviderEvent,
    ProviderFetch,
)
from src.intelligence.calendar_providers.official_sources.base_official import load_catalog
from src.intelligence.calendar_providers.official_sources.organisms import (
    ALL_OFFICIAL_PROVIDERS,
)

logger = logging.getLogger(__name__)


def _served_keys(events: Sequence[ProviderEvent]) -> set:
    """Catalog keys that produced ≥1 dated instance. ``provider_ref`` is
    ``"<event_key>:<date>"`` and event keys never contain ':'."""
    keys = set()
    for e in events:
        ref = getattr(e, "provider_ref", "") or ""
        key = ref.split(":", 1)[0]
        if key:
            keys.add(key)
    return keys


class OfficialCalendarProvider(CalendarProvider):
    """Default provider: composes the per-organism official adapters."""

    def __init__(self, providers: Optional[Sequence[CalendarProvider]] = None) -> None:
        self._providers: Sequence[CalendarProvider] = (
            providers if providers is not None else [p() for p in ALL_OFFICIAL_PROVIDERS]
        )

    @property
    def source_name(self) -> str:
        return "official"

    def fetch(self) -> ProviderFetch:
        all_events = []
        starts = []
        ends = []
        for provider in self._providers:
            try:
                sub = provider.fetch()
            except Exception as exc:  # defensive — one source never sinks the rest
                logger.warning("official sub-provider %s failed: %s", getattr(provider, "source_name", "?"), exc)
                continue
            all_events.extend(sub.events)
            if sub.coverage_start is not None:
                starts.append(sub.coverage_start)
            if sub.coverage_end is not None:
                ends.append(sub.coverage_end)
        self._warn_undatable(all_events)
        return ProviderFetch(
            events=all_events,
            coverage_start=min(starts) if starts else None,
            coverage_end=max(ends) if ends else None,
        )

    def attributions(self) -> List[ProviderAttribution]:
        out: List[ProviderAttribution] = []
        for provider in self._providers:
            out.extend(provider.attributions())
        return out

    def undatable_events(self, events: Optional[Sequence[ProviderEvent]] = None) -> List[str]:
        """Catalog events that produced NO dated instance from any wired source —
        i.e. events that would silently vanish from the calendar (NW-4 ch.5). The
        rule: such an event is SIGNALLED (logged + auditable here), never dropped
        without trace. ``events`` may be passed to avoid a second fetch."""
        served = _served_keys(events if events is not None else self.fetch().events)
        catalog = load_catalog()
        return sorted(k for k in catalog if k not in served)

    def _warn_undatable(self, events: Sequence[ProviderEvent]) -> None:
        undatable = self.undatable_events(events)
        if undatable:
            logger.warning(
                "calendar: %d catalog event(s) produced NO dated instance — "
                "not datable by any wired source (curated schedule or live feed), "
                "surfaced instead of silently dropped: %s",
                len(undatable),
                ", ".join(undatable),
            )


__all__ = ["OfficialCalendarProvider"]
