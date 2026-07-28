"""Economic-calendar schema (NW-1 — "Actualités programmées" list view).

Descriptive/factual only. The calendar announces MOMENTS, never DIRECTIONS
(mission NW-1 §0): no predicted value, no direction, no reaction.

The schema is modelled on the OFFICIAL sources (BLS, BEA, Census, Federal
Reserve, Eurostat, ECB) — NOT on any single feed. It carries, from day one,
the stable recurring identifier, issuing organism, value unit, revisions,
original publication timezone, and per-record source + license. An adapter that
cannot supply a field leaves it ``None``; downstream a ``None`` renders as
ABSENT — never fabricated, never defaulted.

The ForexFactory adapter (prototype, dev-only) leaves most of these ``None`` on
purpose, which makes the gap visible rather than hidden. See
docs/audits/AUDIT-nw-1-calendrier.md for the per-field source comparison.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

CalendarImpact = Literal["high", "medium", "low"]


class CalendarEvent(BaseModel):
    """A single scheduled economic publication attached to ≥1 followed market."""

    # Global dedup id ("<source>:<provider_ref>") — unique across sources, so
    # records from different sources never collide.
    event_id: str
    # Provenance + stable recurring identifier of the indicator.
    source: str
    series_code: Optional[str] = None   # stable recurring code (official) — None for FF
    license_label: Optional[str] = None
    # What the event is.
    event: str
    currency: str
    impact: CalendarImpact
    organism: Optional[str] = None      # issuing organism — None for FF
    # When: UTC + the publisher's own timezone.
    scheduled_at: datetime
    source_timezone: Optional[str] = None
    # Markets this event is attached to (config/event_market_map.json). Never
    # empty: an event attached to no followed market is dropped before serialization.
    markets: List[str] = Field(default_factory=list)
    # Values — in the indicator's own unit, never a price, never converted.
    value_unit: Optional[str] = None
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    revised: bool = False
    previous_before_revision: Optional[float] = None


class CalendarCoverage(BaseModel):
    """Honest coverage of the underlying source vs the requested window.

    ``partial`` is True when the requested window extends beyond the range the
    source actually covers, so the page states it instead of looking complete on
    partial data.
    """

    source: str = "official"
    feed_start: Optional[datetime] = None
    feed_end: Optional[datetime] = None
    partial: bool = False


class CalendarResponse(BaseModel):
    """Chronological list of attached events in a window, plus coverage."""

    events: List[CalendarEvent] = Field(default_factory=list)
    window_start: datetime
    window_end: datetime
    coverage: CalendarCoverage = Field(default_factory=CalendarCoverage)
    generated_at: datetime


__all__ = [
    "CalendarImpact",
    "CalendarEvent",
    "CalendarCoverage",
    "CalendarResponse",
]
