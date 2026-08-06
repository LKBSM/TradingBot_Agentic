/**
 * Engine-measured facts for a recurring publication (NW-3) — mirror of
 * src/intelligence/publication_measures_schema.py.
 *
 * The numbers only: every visible sentence is composed frontend-side from static
 * i18n templates, never generated. Durations are WHOLE MINUTES (never candle
 * counts); the UI formats hours/minutes. A measure that cannot be computed
 * reliably is `null` (omitted) — the page renders nothing for it, never a
 * placeholder. Zone-lifecycle (question #3) is computed since NW-7.
 */

/** One dated extreme of a distribution — a duration (minutes) OR an amplitude. */
export interface MeasureExtreme {
  observed_at: string; // ISO UTC — the publication instant it was observed at
  minutes: number | null;
  amount: number | null;
}

/** One bucket of a duration distribution; bounds in minutes (UI formats h/min). */
export interface DurationTranche {
  lower_minutes: number;
  upper_minutes: number | null; // null ⇒ open upper bound ("beyond")
  count: number;
}

/** The source line every displayed measure carries — method + denominator. */
export interface MeasureProvenance {
  method_key: string;
  sample_size: number;
  market: string;
  period_start: string; // ISO UTC
  period_end: string; // ISO UTC
  reference_days: number | null;
  quote_unit: string | null;
}

export interface CalmBeforeMeasure {
  provenance: MeasureProvenance;
  reference_amount: number;
  calmer_count: number;
  busier_count: number;
  calmest: MeasureExtreme;
  busiest: MeasureExtreme;
}

export interface StructureStateMeasure {
  provenance: MeasureProvenance;
  inside_zone_count: number;
  intact_pocket_count: number;
  range_lower_count: number;
  range_middle_count: number;
  range_upper_count: number;
  now_inside_zone: boolean | null;
  now_intact_pocket_within: boolean | null;
  now_range_position: 'lower' | 'middle' | 'upper' | null;
}

export interface ReturnToCalmMeasure {
  provenance: MeasureProvenance;
  tranches: DurationTranche[];
  fastest: MeasureExtreme;
  slowest: MeasureExtreme;
  never_settled_count: number;
}

export interface ZoneLifecycleMeasure {
  provenance: MeasureProvenance;
  zones_created_count: number;
  tranches: DurationTranche[]; // minutes from a zone's creation to first mitigation
  fastest: MeasureExtreme;
  slowest: MeasureExtreme;
  never_mitigated_count: number;
}

export interface PublicationMeasures {
  event_key: string;
  market: string;
  calm_before: CalmBeforeMeasure | null;
  structure_state: StructureStateMeasure | null;
  zone_lifecycle: ZoneLifecycleMeasure | null;
  return_to_calm: ReturnToCalmMeasure | null;
}

/** True when at least one measure is computed (so the section should render). */
export function hasAnyMeasure(m: PublicationMeasures | null | undefined): boolean {
  return (
    !!m &&
    !!(m.calm_before || m.structure_state || m.zone_lifecycle || m.return_to_calm)
  );
}
