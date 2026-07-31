"""Engine-measured facts around a recurring economic publication (NW-3).

The publication page asks four questions of the PAST releases of one indicator,
measured on ONE attached market. This module is the CONTRACT both the backend
computation and the frontend rendering bind to — the numbers only; every visible
sentence is composed frontend-side from static i18n templates, never generated.

Discipline baked into the shape (mission NW-3 LIGNE + writing rules):
  · No direction, no consensus, no causality — this is descriptive measurement.
  · Durations travel as WHOLE MINUTES (``minutes``), never candle counts; the
    reader did not pick a timeframe on this page, so the UI formats hours/minutes.
  · No central-value statistic is exposed (no median/mean/std). A continuous
    quantity is described by its FULL distribution in tranches plus the dated
    extremes — twelve very different days must not read as one regular number.
  · Every measure carries its ``MeasureProvenance`` — the method named and the
    denominator (how many publications, which market, over which period, and the
    reference sample when one is used). A measure without it is a bug tests catch.
  · A measure that cannot be computed RELIABLY is OMITTED (``None``) — never
    approximated, never a placeholder promising future content (§2 periscope).

Zone-lifecycle (formation→mitigation, question #3) is DEFERRED this iteration and
is intentionally ABSENT from the model: no field advertises content that is not
computed.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class MeasureExtreme(BaseModel):
    """One dated extreme of a distribution (calmest/busiest, fastest/slowest).

    Carries the publication instant it was observed at, plus EITHER a duration
    (``minutes``, whole minutes — never candles) OR a price amplitude
    (``amount``, in the market's own quote unit, never converted)."""

    observed_at: datetime
    minutes: Optional[int] = None
    amount: Optional[float] = None


class DurationTranche(BaseModel):
    """One bucket of a duration distribution. Bounds in whole minutes; the UI
    formats them to hours/minutes. An open upper bound is ``None`` ("beyond")."""

    lower_minutes: int
    upper_minutes: Optional[int] = None
    count: int


class MeasureProvenance(BaseModel):
    """The source line every displayed measure MUST carry.

    Names the method and the denominator so the reader always knows the base of
    the count: how many publications were measured, on which single market, over
    which period, and — when the measure compares to a baseline — how many
    non-publication reference days define it."""

    method_key: str                       # static i18n id naming the method
    sample_size: int                      # denominator — publications measured
    market: str                           # the single market measured
    period_start: datetime                # first publication in the sample
    period_end: datetime                  # last publication in the sample
    reference_days: Optional[int] = None  # non-publication days defining the baseline
    quote_unit: Optional[str] = None      # how ``amount`` reads (e.g. "USD")


class CalmBeforeMeasure(BaseModel):
    """#1 — the hour BEFORE each publication compared to the same hour-of-day on
    the reference (non-publication) days. Facts and counts, not a central value.

    ``reference_amount`` is the ordinary amplitude for that hour-of-day, defined
    on screen so the comparison is never against an undefined reference."""

    provenance: MeasureProvenance
    reference_amount: float
    calmer_count: int   # publications quieter than the ordinary hour
    busier_count: int   # publications busier than the ordinary hour
    calmest: MeasureExtreme
    busiest: MeasureExtreme


class StructureStateMeasure(BaseModel):
    """#2 — the structural context at the instant of each publication, obtained by
    REPLAYING detection on the window ending at that instant (no look-ahead).

    Counts across the sample, plus the state detected NOW before the upcoming
    release (to situate the moment, never to anticipate the reaction). Range
    position is bucketed into thirds — a factual band, not a score."""

    provenance: MeasureProvenance
    inside_zone_count: int        # price already inside an OB/FVG zone at T
    intact_pocket_count: int      # an intact liquidity pocket within 0.5% of price at T
    range_lower_count: int
    range_middle_count: int
    range_upper_count: int
    now_inside_zone: Optional[bool] = None
    now_intact_pocket_within: Optional[bool] = None
    now_range_position: Optional[str] = None  # "lower" | "middle" | "upper"


class ReturnToCalmMeasure(BaseModel):
    """#4 — minutes after each publication until the per-bar movement returns to
    the hour's ordinary level. FULL distribution in tranches + dated extremes;
    publications that never settle within the observed window are counted apart."""

    provenance: MeasureProvenance
    tranches: List[DurationTranche] = Field(default_factory=list)
    fastest: MeasureExtreme
    slowest: MeasureExtreme
    never_settled_count: int = 0


class PublicationMeasures(BaseModel):
    """Engine-measured facts for ONE recurring publication on ONE attached market.

    Each measure is ``None`` when it cannot be computed reliably — the page then
    simply does not render it (never a placeholder). ``has_any`` lets the endpoint
    decide whether the measures section exists at all."""

    event_key: str
    market: str
    calm_before: Optional[CalmBeforeMeasure] = None
    structure_state: Optional[StructureStateMeasure] = None
    return_to_calm: Optional[ReturnToCalmMeasure] = None

    def has_any(self) -> bool:
        return any(
            m is not None
            for m in (self.calm_before, self.structure_state, self.return_to_calm)
        )


__all__ = [
    "MeasureExtreme",
    "DurationTranche",
    "MeasureProvenance",
    "CalmBeforeMeasure",
    "StructureStateMeasure",
    "ReturnToCalmMeasure",
    "PublicationMeasures",
]
