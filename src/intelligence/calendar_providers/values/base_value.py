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
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_ENV_VALUES_LIVE = "CALENDAR_VALUES_LIVE"


@dataclass(frozen=True)
class ValuePoint:
    """A fetched value for a release — AS PUBLISHED, never converted/rounded."""

    actual: float
    previous: Optional[float] = None


@dataclass(frozen=True)
class SeriesPoint:
    """One observation in a published series — the reference PERIOD label (e.g.
    "2026-06", as the organism labels it) and the values AS PUBLISHED. The period
    is the indicator's own reference month/quarter, NOT a release date.

    ``value`` is the PRIMARY value plotted on the curve: the headline VARIATION when
    the organism publishes one (a % change, or an absolute change for a count),
    else the raw level. ``level`` is the raw official level kept for the second
    plan / hover when ``value`` is a variation derived from it (an index level, a
    total count) — ``None`` when the published value already IS a variation.
    ``change_mom`` is the SECONDARY month-over-month % shown alongside a headline
    annual % (index series) — ``None`` otherwise. Everything AS PUBLISHED."""

    period: str
    value: float
    level: Optional[float] = None
    change_mom: Optional[float] = None


def derive_variation_series(points: "List[SeriesPoint]", mode: str) -> "List[SeriesPoint]":
    """Compute a VARIATION series from a series of published LEVELS (NW-8 Batch 2).

    The organism publishes only the level (a price index, a dollar amount) and not
    the change, so the product computes it — EXACTLY, from two published levels of
    the SAME (seasonally-adjusted) series — and the caller attributes it as
    "computed", never "published". No value is fabricated: a point without the
    levels its change needs is dropped.

    ``mode``:
      · "index"  — an index: ``value`` = the 12-month % change (headline),
        ``level`` = the index, ``change_mom`` = the 1-month % change. Points before
        the 12th are dropped (no year-ago level).
      · "amount" — a dollar amount / count: ``value`` = the 1-month % change
        (headline), ``level`` = the amount. The first point is dropped.

    ``points`` are the level series oldest→newest (``value`` holding the level)."""
    levels = [p.value for p in points]
    out: List[SeriesPoint] = []
    for i, p in enumerate(points):
        lvl = levels[i]
        mom = ((lvl / levels[i - 1] - 1.0) * 100.0) if i >= 1 and levels[i - 1] else None
        if mode == "index":
            if i < 12 or not levels[i - 12]:
                continue  # no year-ago level → no 12-month change → no point
            yoy = (lvl / levels[i - 12] - 1.0) * 100.0
            out.append(SeriesPoint(period=p.period, value=yoy, level=lvl, change_mom=mom))
        else:  # "amount"
            if mom is None:
                continue  # no prior month → no 1-month change → no point
            out.append(SeriesPoint(period=p.period, value=mom, level=lvl))
    return out


class ValueFetcher(ABC):
    """Fetches the published value for one series of one source."""

    @abstractmethod
    def fetch(self, series_code: str) -> Optional[ValuePoint]:
        """Return the latest published value (+ previous) for ``series_code``, or
        ``None`` on any failure/absence (never raises, never fabricates)."""

    def fetch_series(
        self, series_code: str, limit: int = 12, kind: str = "level"
    ) -> List[SeriesPoint]:
        """Return the last ``limit`` published observations (oldest→newest) for
        ``series_code``, each with its reference-period label. ``kind`` selects how
        the published series should read: "level" (the value as stored) or
        "yoy_percent" (the source's own 12-month percent change, when it publishes
        one). A fetcher that does not honour a ``kind`` may ignore it. Default:
        empty — a source that cannot serve a series (or has none) returns [] and
        the page simply shows no curve. Never fabricates, never raises."""
        return []


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

    def series_for(
        self,
        source: str,
        series_code: Optional[str],
        limit: int = 12,
        kind: str = "level",
    ) -> List[SeriesPoint]:
        """Route a series-of-observations request to its source's fetcher. A
        source with no registered fetcher, or one that serves no series, yields
        [] (→ no curve, honestly). ``kind`` ("level" | "yoy_percent") selects how
        the series reads. Never raises, never fabricates."""
        if not series_code:
            return []
        fetcher = self._by_source.get(source)
        if fetcher is None:
            return []
        try:
            return fetcher.fetch_series(series_code, limit, kind)
        except Exception as exc:  # defensive — a fetch failure is graceful
            logger.warning("value series fetch failed for %s/%s: %s", source, series_code, exc)
            return []

    def fetch(self, series_code: str) -> Optional[ValuePoint]:  # pragma: no cover
        raise NotImplementedError("use fetch_for(source, series_code)")


def build_value_fetcher() -> Optional[MultiValueFetcher]:
    """Build the configured value fetcher, or ``None`` when live values are OFF.

    Default OFF (deterministic tests / stable default). With ``CALENDAR_VALUES_LIVE=1``
    the no-key sources are wired live (ECB, Eurostat); key-gated sources
    (BLS/BEA/Census) register only when their key env is present — otherwise they
    stay absent and their events remain honestly ``unfetched``.
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
    if os.environ.get("BEA_API_KEY"):
        from src.intelligence.calendar_providers.values.bea_values import BEAValueFetcher

        by_source["bea"] = BEAValueFetcher()
    if os.environ.get("CENSUS_API_KEY"):
        from src.intelligence.calendar_providers.values.census_values import (
            CensusValueFetcher,
        )

        by_source["census"] = CensusValueFetcher()
    return MultiValueFetcher(by_source)


__all__ = [
    "ValuePoint",
    "SeriesPoint",
    "ValueFetcher",
    "MultiValueFetcher",
    "build_value_fetcher",
]
