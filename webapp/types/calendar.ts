/**
 * Economic-calendar types (NW-1 / NW-1b) — mirror of
 * src/intelligence/calendar_schema.py.
 *
 * Descriptive only: a scheduled MOMENT plus, per record, its source, organism,
 * periodicity, unit and revision history — never a predicted value, direction or
 * reaction. NW-1b removed ``impact`` (no organism ranks its releases) and
 * ``forecast`` (no organism publishes an analyst consensus). A field the source
 * does not provide is `null` and is rendered as ABSENT, never defaulted.
 */

export type CalendarPeriodicity = 'monthly' | 'quarterly' | 'eight_per_year';

export interface CalendarEvent {
  event_id: string;
  source: string;
  series_code: string | null;
  license_label: string | null;
  event: string;
  currency: string;
  organism: string | null;
  periodicity: CalendarPeriodicity | null;
  scheduled_at: string; // ISO UTC
  source_timezone: string | null;
  time_confirmed: boolean;
  markets: string[];
  value_unit: string | null;
  actual: number | null;
  actual_initial: number | null;
  previous: number | null;
  revised: boolean;
  revised_at: string | null; // ISO UTC
}

export interface CalendarCoverage {
  source: string;
  feed_start: string | null;
  feed_end: string | null;
  partial: boolean;
  last_success: Record<string, string>; // source → ISO UTC of last successful refresh
  stale_sources: string[];
}

export interface CalendarAttribution {
  source: string;
  organism: string;
  license_label: string;
  policy_url: string;
}

export interface CalendarResponse {
  events: CalendarEvent[];
  window_start: string;
  window_end: string;
  coverage: CalendarCoverage;
  attribution: CalendarAttribution[];
  generated_at: string;
}
