"""Published-value enrichment for calendar events (NW-1c §3A).

The calendar (NW-1/1b) provides DATES + metadata; this package fetches the
PUBLISHED VALUE + previous value by STABLE SERIES CODE (never by title). Each
organism has its own fetcher behind :class:`ValueFetcher`; a source with no
fetcher (or an unreachable one) leaves the value absent → the event stays in the
honest ``unfetched`` state (mission §3B), never a fabricated number.

Opt-in via ``CALENDAR_VALUES_LIVE=1`` (default OFF for deterministic tests and a
stable default deployment). A fetch never destroys stored data.
"""

from src.intelligence.calendar_providers.values.base_value import (
    MultiValueFetcher,
    SeriesPoint,
    ValueFetcher,
    ValuePoint,
    build_value_fetcher,
    diagnose_value_fetcher,
)

__all__ = [
    "ValueFetcher",
    "ValuePoint",
    "SeriesPoint",
    "MultiValueFetcher",
    "build_value_fetcher",
    "diagnose_value_fetcher",
]
