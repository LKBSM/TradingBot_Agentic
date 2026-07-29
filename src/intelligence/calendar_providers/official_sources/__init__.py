"""Per-organism official calendar adapters (NW-1b).

One adapter per issuing organism (BLS, BEA, Census, Federal Reserve, Eurostat,
ECB), all sharing :class:`OfficialSourceProvider` and differing only by their
source key + (future) live feed. They are composed by
``OfficialCalendarProvider`` (the default source) so the service sees a single
provider while every event keeps its own source, organism and licence.
"""

from src.intelligence.calendar_providers.official_sources.base_official import (
    OfficialSourceProvider,
    ReleaseInstance,
    load_catalog,
    load_schedule,
)

__all__ = [
    "OfficialSourceProvider",
    "ReleaseInstance",
    "load_catalog",
    "load_schedule",
]
