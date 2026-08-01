import type { CalendarEvent } from '@/types/calendar';
import { parseUtc } from '@/lib/time/localTime';

/**
 * Pure grouping / time helpers for the calendar list (NW-1).
 *
 * The ONLY ordering is chronological — there is deliberately NO amplitude / past
 * -magnitude sort (that would rank future events by supposed interest, mission
 * §1B). Events are grouped by calendar day in the reader's timezone.
 */

export interface CountdownParts {
  past: boolean;
  totalMinutes: number;
  days: number;
  hours: number;
  minutes: number;
}

/** Signed countdown between `now` and a scheduled instant. */
export function countdown(nowMs: number, whenMs: number): CountdownParts {
  const diffMin = Math.round((whenMs - nowMs) / 60_000);
  const past = diffMin < 0;
  const abs = Math.abs(diffMin);
  return {
    past,
    totalMinutes: diffMin,
    days: Math.floor(abs / 1440),
    hours: Math.floor((abs % 1440) / 60),
    minutes: abs % 60,
  };
}

/** YYYY-MM-DD in the given (or local) timezone — the day-grouping key. */
export function dayKey(d: Date, timeZone?: string): string {
  // en-CA yields ISO-like YYYY-MM-DD; timeZone shifts to the reader's day.
  return d.toLocaleDateString('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
}

export interface DayGroup {
  key: string;
  /** A representative instant inside the day (first event), for labelling. */
  date: Date;
  isToday: boolean;
  isTomorrow: boolean;
  events: CalendarEvent[];
}

/**
 * Split events into chronological day groups (reader's timezone). Assumes the
 * input is already the window the caller wants (past/upcoming filtering is done
 * upstream). Order within and across groups is strictly by time.
 */
export function groupEventsByDay(
  events: CalendarEvent[],
  now: Date,
  timeZone?: string,
): DayGroup[] {
  const sorted = [...events].sort(
    (a, b) => toMs(a.scheduled_at) - toMs(b.scheduled_at),
  );
  const todayKey = dayKey(now, timeZone);
  const tomorrow = new Date(now.getTime() + 86_400_000);
  const tomorrowKey = dayKey(tomorrow, timeZone);

  const groups = new Map<string, DayGroup>();
  for (const ev of sorted) {
    const d = parseUtc(ev.scheduled_at);
    if (!d) continue;
    const key = dayKey(d, timeZone);
    let g = groups.get(key);
    if (!g) {
      g = {
        key,
        date: d,
        isToday: key === todayKey,
        isTomorrow: key === tomorrowKey,
        events: [],
      };
      groups.set(key, g);
    }
    g.events.push(ev);
  }
  return [...groups.values()];
}

/** Long, localized day label — « mardi 28 juillet » / « Tuesday, July 28 ». */
export function longDayLabel(d: Date, locale: string, timeZone?: string): string {
  return d.toLocaleDateString(locale, {
    timeZone,
    weekday: 'long',
    day: 'numeric',
    month: 'long',
  });
}

/** « 09:30 » in a specific timezone (market/publisher time). */
export function hmInZone(d: Date, timeZone?: string): string {
  return d.toLocaleTimeString('fr-FR', {
    timeZone,
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Factual multi-facet filter (NW-1b): keep events whose SOURCE is in `sources`,
 * that are attached to a MARKET in `markets`, and whose PERIODICITY is in
 * `periodicities`. All three are factual — never a rank. Empty sets keep nothing
 * (the caller shows the explicit empty message, never a silent fall-back).
 *
 * Periodicity is the only optional facet: a record the source does not tag with
 * a cadence (`periodicity == null`, e.g. the dev prototype) is not filterable on
 * that facet and passes it — organism and market stay strict.
 */
export function filterEvents(
  events: CalendarEvent[],
  sources: ReadonlySet<string>,
  markets: ReadonlySet<string>,
  periodicities: ReadonlySet<string>,
): CalendarEvent[] {
  return events.filter(
    (e) =>
      sources.has(e.source) &&
      e.markets.some((m) => markets.has(m)) &&
      (e.periodicity == null || periodicities.has(e.periodicity)),
  );
}

/** Split into past vs upcoming relative to `now` (both chronological). */
export function splitPastUpcoming(
  events: CalendarEvent[],
  now: Date,
): { past: CalendarEvent[]; upcoming: CalendarEvent[] } {
  const nowMs = now.getTime();
  const past: CalendarEvent[] = [];
  const upcoming: CalendarEvent[] = [];
  for (const e of events) {
    if (toMs(e.scheduled_at) < nowMs) past.push(e);
    else upcoming.push(e);
  }
  return { past, upcoming };
}

function toMs(iso: string): number {
  return parseUtc(iso)?.getTime() ?? 0;
}

/**
 * The most recent successful refresh across all sources (NW-4 ch.5): the max of
 * the per-source `last_success` map. Returned so a calendar view can show the
 * PROOF of freshness permanently — not just a stale marker when something breaks.
 * Null when no success is recorded yet (the view then shows nothing rather than a
 * fabricated date).
 */
export function latestSuccess(
  lastSuccess: Record<string, string> | null | undefined,
): Date | null {
  if (!lastSuccess) return null;
  let best = 0;
  for (const iso of Object.values(lastSuccess)) {
    const ms = parseUtc(iso)?.getTime() ?? 0;
    if (ms > best) best = ms;
  }
  return best > 0 ? new Date(best) : null;
}
