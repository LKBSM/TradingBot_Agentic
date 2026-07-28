"""Official-source calendar provider — the DEFAULT source (NW-1).

The target source of truth is the official statistical organisms (BLS, BEA,
Census, Federal Reserve for the US; Eurostat, ECB for the euro area): free,
public-domain or permissively reusable, with stable identifiers, issuing
organism, value unit and revisions.

Integrating those feeds is OUT OF SCOPE for NW-1 — it is a dedicated follow-up
mission. This provider is therefore a deliberate STUB: it returns no events, so
by default the calendar page is honestly empty (source + organism columns
visibly blank) rather than showing prototype data the product has no right to
display. NW-1's job is only to make that follow-up cost "just an adapter".
"""

from __future__ import annotations

import logging

from src.intelligence.calendar_providers.base import CalendarProvider, ProviderFetch

logger = logging.getLogger(__name__)


class OfficialCalendarProvider(CalendarProvider):
    """Default provider. Yields nothing until the official-source integration
    mission wires the real organism feeds behind this same interface."""

    _warned = False

    @property
    def source_name(self) -> str:
        return "official"

    def fetch(self) -> ProviderFetch:
        if not OfficialCalendarProvider._warned:
            logger.info(
                "OfficialCalendarProvider is a stub (official-source integration "
                "is a follow-up mission, out of NW-1 scope) — returning 0 events. "
                "Set CALENDAR_SOURCE=forexfactory for local prototype data."
            )
            OfficialCalendarProvider._warned = True
        return ProviderFetch(events=[], coverage_start=None, coverage_end=None)


__all__ = ["OfficialCalendarProvider"]
