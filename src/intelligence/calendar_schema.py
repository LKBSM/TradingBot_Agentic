"""Economic-calendar schema (NW-1 / NW-1b — official-source integration).

Descriptive/factual only. The calendar announces MOMENTS, never DIRECTIONS
(mission NW-1 §0): no predicted value, no direction, no reaction.

The schema is modelled on the OFFICIAL sources (BLS, BEA, Census, Federal
Reserve, Eurostat, ECB) — NOT on any single feed. It carries, per record: the
stable recurring identifier, the issuing organism, the publication periodicity,
the value unit, the revision history (initial vs current value + revision date),
the original publication timezone, and the source + license label.

NW-1b removed two fields the reality of the sources does not support:
  · NO ``forecast`` / consensus — no public organism publishes an analyst
    forecast; a private paid survey is not the product's to display or fabricate.
  · NO ``impact`` ranking — no organism grades its releases high/medium/low; a
    ranking would be a conviction score, which the product forbids everywhere.
Filtering is therefore factual: issuing organism, attached market, periodicity.

An adapter that cannot supply a field leaves it ``None``; downstream a ``None``
renders as ABSENT — never fabricated, never defaulted. See
docs/audits/AUDIT-nw-1b-sources-officielles.md for the per-field source table.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# Factual publication cadence — a fact of the release, NOT a rank.
CalendarPeriodicity = Literal["monthly", "quarterly", "eight_per_year"]


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
    organism: Optional[str] = None      # issuing organism — None for FF
    periodicity: Optional[CalendarPeriodicity] = None
    # When: UTC + the publisher's own timezone.
    scheduled_at: datetime
    source_timezone: Optional[str] = None
    # Whether the publication TIME (not just the date) is confirmed from an
    # official source. A publication whose time is not confirmed is never
    # approximated and is EXCLUDED from any amplitude measure (NW-2).
    time_confirmed: bool = True
    # Markets this event is attached to (config/event_market_map.json). Never
    # empty: an event attached to no followed market is dropped before serialization.
    markets: List[str] = Field(default_factory=list)
    # Values — in the indicator's own unit, never a price, never converted.
    value_unit: Optional[str] = None
    actual: Optional[float] = None          # current (possibly revised) value
    actual_initial: Optional[float] = None  # value FIRST published for this release
    previous: Optional[float] = None        # prior period's published value (factual)
    revised: bool = False
    revised_at: Optional[datetime] = None   # when ``actual`` was revised, if it was


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
    # Per-source freshness: the last time each source was successfully refreshed.
    # A source that failed to refresh keeps its stored data and its prior
    # timestamp here — the page shows "not refreshed since <date>", never blanks.
    last_success: Dict[str, datetime] = Field(default_factory=dict)
    stale_sources: List[str] = Field(default_factory=list)


class CalendarAttribution(BaseModel):
    """One entry per source actually used, naming it and linking its reuse
    policy. Rendering an event without its attribution is a licence breach, so
    this travels with every response and is asserted in tests."""

    source: str
    organism: str
    license_label: str
    policy_url: str


class CalendarResponse(BaseModel):
    """Chronological list of attached events in a window, plus coverage and the
    per-source attribution block (licence condition)."""

    events: List[CalendarEvent] = Field(default_factory=list)
    window_start: datetime
    window_end: datetime
    coverage: CalendarCoverage = Field(default_factory=CalendarCoverage)
    attribution: List[CalendarAttribution] = Field(default_factory=list)
    generated_at: datetime


__all__ = [
    "CalendarPeriodicity",
    "CalendarEvent",
    "CalendarCoverage",
    "CalendarAttribution",
    "CalendarResponse",
]
