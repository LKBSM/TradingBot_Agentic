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

# The distinct states of a value. A bare "—" reads three ways and a client can't
# tell them apart, so the three ABSENCES are modelled explicitly (mission NW-1c
# §3B), never collapsed to null:
#   · published   — the value was obtained and is shown.
#   · pending     — the publication date is in the future; it does not exist yet.
#   · unfetched   — the publication happened but the value was NOT obtained. A
#                   PRODUCT state (not a market state), shown with the last attempt.
#   · unavailable — the source does not expose a fetchable value for this release
#                   (e.g. a rate decision with no data series). Named organism.
ValueState = Literal["published", "pending", "unfetched", "unavailable"]


def compute_value_state(
    series_code: Optional[str],
    actual: Optional[float],
    scheduled_at: datetime,
    now: datetime,
) -> ValueState:
    """Classify an ``actual`` value into one of the four states. Order matters:
    a future release is ``pending`` even without a series (the value cannot exist
    yet); a past release with no series linkage is ``unavailable`` (nothing to
    fetch); a past release with a series but no value is ``unfetched``."""
    if actual is not None:
        return "published"
    if scheduled_at > now:
        return "pending"
    if not series_code:
        return "unavailable"
    return "unfetched"


class CalendarSeriesPoint(BaseModel):
    """One observation in an indicator's published history — the reference PERIOD
    label (as the organism labels it, e.g. "2026-06") and the values AS PUBLISHED.
    The period is the indicator's own reference month/quarter, never a release
    date and never converted. Feeds the detail page's curve.

    ``value`` is the PRIMARY plotted value: the headline VARIATION when the organism
    publishes one (% change, or an absolute change for a count), else the raw level.
    ``level`` is the raw official level kept for the second plan / hover (index
    level, total count) — ``None`` when the value already IS a variation.
    ``change_mom`` is the SECONDARY month-over-month % shown next to a headline
    annual % — ``None`` otherwise."""

    period: str
    value: float
    level: Optional[float] = None
    change_mom: Optional[float] = None


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
    # How the published series READS on the curve (NW-8): "index_change" (an index
    # level whose headline is a % change), "count_change" (a count whose headline
    # is an absolute monthly change), "published_change" (the value already IS a
    # variation). None ⇒ the raw level is shown as-is (no variation). Drives the
    # frontend's variation-first rendering + its unit labels.
    variation_kind: Optional[str] = None
    # True when the displayed variation is PUBLISHED by the organism (vs computed
    # by the product) — the attribution line must never blur the two.
    variation_published: bool = True
    actual: Optional[float] = None          # current (possibly revised) value
    actual_initial: Optional[float] = None  # value FIRST published for this release
    previous: Optional[float] = None        # prior period's published value (factual)
    revised: bool = False
    revised_at: Optional[datetime] = None   # when ``actual`` was revised, if it was
    # The explicit value state (see ``compute_value_state``) — never a bare null.
    actual_state: ValueState = "pending"
    # Last time this event was refreshed / a value fetch was attempted (the
    # "last attempt" date shown for the ``unfetched`` state).
    refreshed_at: Optional[datetime] = None
    # The indicator's published history (oldest→newest), when the source serves a
    # series. Empty for events with no series or no live value flow — the detail
    # page then shows no curve (never a fabricated one). Attached ONLY by the
    # per-event detail path, never by the list/month window (one call per detail).
    value_series: List["CalendarSeriesPoint"] = Field(default_factory=list)


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
    "ValueState",
    "compute_value_state",
    "CalendarSeriesPoint",
    "CalendarEvent",
    "CalendarCoverage",
    "CalendarAttribution",
    "CalendarResponse",
]
