/**
 * Economic-calendar types (NW-1) — mirror of src/intelligence/calendar_schema.py.
 *
 * Descriptive only: a scheduled MOMENT plus, per record, its source + organism
 * when known — never a predicted value, direction or reaction. A field the
 * source does not provide is `null` and is rendered as ABSENT, never defaulted.
 */

export type CalendarImpact = 'high' | 'medium' | 'low';

export interface CalendarEvent {
  event_id: string;
  source: string;
  series_code: string | null;
  license_label: string | null;
  event: string;
  currency: string;
  impact: CalendarImpact;
  organism: string | null;
  scheduled_at: string; // ISO UTC
  source_timezone: string | null;
  markets: string[];
  value_unit: string | null;
  actual: number | null;
  forecast: number | null;
  previous: number | null;
  revised: boolean;
  previous_before_revision: number | null;
}

export interface CalendarCoverage {
  source: string;
  feed_start: string | null;
  feed_end: string | null;
  partial: boolean;
}

export interface CalendarResponse {
  events: CalendarEvent[];
  window_start: string;
  window_end: string;
  coverage: CalendarCoverage;
  generated_at: string;
}
