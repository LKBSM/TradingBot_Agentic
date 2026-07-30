"""Value-fetch interface + dispatcher (NW-1c §3A).

A ``ValueFetcher`` returns the published value (+ previous) for ONE stable series
code, or ``None`` when it cannot (unreachable, no data, or not implemented for
that source). The dispatcher routes by source to the per-organism fetcher.

No value is ever fabricated: a ``None`` keeps the event in the ``unfetched``
state. The linkage is by series code — never by title.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_ENV_VALUES_LIVE = "CALENDAR_VALUES_LIVE"


@dataclass(frozen=True)
class ValuePoint:
    """A fetched value for a release — AS PUBLISHED, never converted/rounded."""

    actual: float
    previous: Optional[float] = None


class ValueFetcher(ABC):
    """Fetches the published value for one series of one source."""

    @abstractmethod
    def fetch(self, series_code: str) -> Optional[ValuePoint]:
        """Return the latest published value (+ previous) for ``series_code``, or
        ``None`` on any failure/absence (never raises, never fabricates)."""


class MultiValueFetcher(ValueFetcher):
    """Routes a series to its source's fetcher. A source with no registered
    fetcher yields ``None`` (→ the event stays ``unfetched``, honestly)."""

    def __init__(self, by_source: Dict[str, ValueFetcher]) -> None:
        self._by_source = by_source

    def fetch_for(self, source: str, series_code: Optional[str]) -> Optional[ValuePoint]:
        if not series_code:
            return None
        fetcher = self._by_source.get(source)
        if fetcher is None:
            return None
        try:
            return fetcher.fetch(series_code)
        except Exception as exc:  # defensive — a fetch failure is graceful
            logger.warning("value fetch failed for %s/%s: %s", source, series_code, exc)
            return None

    def fetch(self, series_code: str) -> Optional[ValuePoint]:  # pragma: no cover
        raise NotImplementedError("use fetch_for(source, series_code)")


def build_value_fetcher() -> Optional[MultiValueFetcher]:
    """Build the configured value fetcher, or ``None`` when live values are OFF.

    Default OFF (deterministic tests / stable default). With ``CALENDAR_VALUES_LIVE=1``
    the no-key sources are wired live (ECB); key-gated sources (BEA/BLS/Census)
    register only when their key env is present — otherwise they stay absent and
    their events remain honestly ``unfetched``.
    """
    if os.environ.get(_ENV_VALUES_LIVE, "").strip().lower() not in ("1", "true", "yes"):
        return None
    from src.intelligence.calendar_providers.values.ecb_values import ECBValueFetcher
    from src.intelligence.calendar_providers.values.eurostat_values import (
        EurostatValueFetcher,
    )

    # No-key sources are always wired when live values are on.
    by_source: Dict[str, ValueFetcher] = {
        "ecb": ECBValueFetcher(),
        "eurostat": EurostatValueFetcher(),
    }
    # Key-gated sources register ONLY when their free API key env is present, so
    # no half-configured call is ever made; otherwise their events stay unfetched.
    if os.environ.get("BLS_API_KEY"):
        from src.intelligence.calendar_providers.values.bls_values import BLSValueFetcher

        by_source["bls"] = BLSValueFetcher()
    return MultiValueFetcher(by_source)


__all__ = ["ValuePoint", "ValueFetcher", "MultiValueFetcher", "build_value_fetcher"]
