"""Calendar provider adapters (NW-1).

The ONLY folder allowed to know a concrete calendar data format. Everything
else (service, store, schema, front) depends solely on the ``CalendarProvider``
interface and neutral ``ProviderEvent``.

Source selection is env-driven and DEFAULTS TO THE OFFICIAL STUB — never to the
ForexFactory prototype:

    CALENDAR_SOURCE=official      (default) → OfficialCalendarProvider (stub, 0 events)
    CALENDAR_SOURCE=forexfactory            → ForexFactoryCalendarProvider (dev only)
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


def build_calendar_provider(source: Optional[str] = None) -> CalendarProvider:
    """Instantiate the configured calendar provider. Default = official stub.

    The ForexFactory prototype is only ever returned when explicitly requested
    via ``CALENDAR_SOURCE=forexfactory`` (local development).
    """
    name = (source or os.environ.get(CALENDAR_SOURCE_ENV_VAR) or DEFAULT_CALENDAR_SOURCE)
    name = name.strip().lower()

    if name == "forexfactory":
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
            "Unknown CALENDAR_SOURCE=%r — falling back to the official stub.", name
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
]
