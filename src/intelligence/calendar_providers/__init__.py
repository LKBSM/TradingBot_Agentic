"""Calendar provider adapters (NW-1).

The ONLY folder allowed to know a concrete calendar data format. Everything
else (service, store, schema, front) depends solely on the ``CalendarProvider``
interface and neutral ``ProviderEvent``.

Source selection is env-driven and DEFAULTS TO THE OFFICIAL STUB — never to the
ForexFactory prototype:

    CALENDAR_SOURCE=official      (default) → OfficialCalendarProvider (stub, 0 events)
    CALENDAR_SOURCE=forexfactory            → ForexFactoryCalendarProvider (dev only,
                                              AND only with CALENDAR_ALLOW_DEV_SOURCE=1)

Production whitelist (CAL-1): only ``OFFICIAL_SOURCES`` may be served. ForexFactory
is a private aggregator, not an issuing organism; selecting it in production
(without the explicit ``CALENDAR_ALLOW_DEV_SOURCE`` dev opt-in) is REFUSED and the
official aggregator is served instead — the guarantee is explicit, not implicit.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from src.intelligence.calendar_providers.base import (
    CalendarProvider,
    ProviderEvent,
    ProviderFetch,
)
from src.intelligence.calendar_providers.official_provider import (
    OfficialCalendarProvider,
)

logger = logging.getLogger(__name__)

DEFAULT_CALENDAR_SOURCE = "official"
CALENDAR_SOURCE_ENV_VAR = "CALENDAR_SOURCE"
# Explicit dev opt-in required to serve the private ForexFactory prototype. Absent
# it, production can never surface a non-official source (CAL-1).
ALLOW_DEV_SOURCE_ENV_VAR = "CALENDAR_ALLOW_DEV_SOURCE"

# The only issuing organisms allowed in production. Kept in lockstep with the
# front-end whitelist (webapp/lib/calendar/officialSources.ts) and the per-organism
# adapters in official_sources/. ForexFactory is deliberately absent.
OFFICIAL_SOURCES = ("bls", "bea", "census", "federal_reserve", "eurostat", "ecb")


def _dev_source_allowed() -> bool:
    return os.environ.get(ALLOW_DEV_SOURCE_ENV_VAR, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def build_calendar_provider(source: Optional[str] = None) -> CalendarProvider:
    """Instantiate the configured calendar provider. Default = official aggregator.

    The ForexFactory prototype is only ever returned when explicitly requested
    via ``CALENDAR_SOURCE=forexfactory`` AND the dev opt-in
    ``CALENDAR_ALLOW_DEV_SOURCE=1`` is set (local development). In production the
    dev flag is unset, so a stray ``CALENDAR_SOURCE=forexfactory`` is REFUSED and
    the official aggregator is served instead — official sources only (CAL-1).
    """
    name = (source or os.environ.get(CALENDAR_SOURCE_ENV_VAR) or DEFAULT_CALENDAR_SOURCE)
    name = name.strip().lower()

    if name == "forexfactory":
        if not _dev_source_allowed():
            logger.error(
                "CALENDAR_SOURCE=forexfactory REFUSED — ForexFactory is a private "
                "aggregator, not an official issuing organism, and has no "
                "commercial display rights. Set CALENDAR_ALLOW_DEV_SOURCE=1 for "
                "local development only. Serving the official aggregator instead."
            )
            return OfficialCalendarProvider()
        # Imported lazily so the dev-only adapter (and its FF coupling) is never
        # loaded in a default/production boot.
        from src.intelligence.calendar_providers.forexfactory_provider import (
            ForexFactoryCalendarProvider,
        )

        logger.warning(
            "CALENDAR_SOURCE=forexfactory — PROTOTYPE adapter (no commercial "
            "display rights). Do NOT enable in a client-facing deployment."
        )
        return ForexFactoryCalendarProvider()

    if name != DEFAULT_CALENDAR_SOURCE:
        logger.warning(
            "Unknown CALENDAR_SOURCE=%r — falling back to the official aggregator.",
            name,
        )
    return OfficialCalendarProvider()


__all__ = [
    "CalendarProvider",
    "ProviderEvent",
    "ProviderFetch",
    "build_calendar_provider",
    "OfficialCalendarProvider",
    "DEFAULT_CALENDAR_SOURCE",
    "CALENDAR_SOURCE_ENV_VAR",
    "ALLOW_DEV_SOURCE_ENV_VAR",
    "OFFICIAL_SOURCES",
]
