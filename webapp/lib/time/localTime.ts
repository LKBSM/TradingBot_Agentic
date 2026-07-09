/**
 * Local-time display helpers.
 *
 * The engine authors EVERY timestamp in UTC (see market_reading_assembler —
 * naive timestamps are treated as UTC, epoch seconds are UTC). The UI, however,
 * should show times in the reader's OWN timezone so « 14:30 » is never ambiguous.
 * These helpers convert a UTC instant to the browser's local timezone and expose
 * one short, stable label for it.
 *
 * Determinism note: output depends on the runtime timezone by design. Callers
 * that need a fixed timezone (tests) pass `timeZone` explicitly.
 */

/** Parse an ISO string the backend authored in UTC. When it carries no offset
 *  we append `Z` so it is read as UTC (not as the runtime-local wall-clock). */
export function parseUtc(iso: string | null | undefined): Date | null {
  if (!iso) return null;
  const s = iso.trim();
  const hasTz = /(?:[zZ]|[+-]\d{2}:?\d{2})$/.test(s);
  const d = new Date(hasTz ? s : `${s}Z`);
  return Number.isNaN(d.getTime()) ? null : d;
}

const HM_OPTS: Intl.DateTimeFormatOptions = { hour: '2-digit', minute: '2-digit' };
const DAY_OPTS: Intl.DateTimeFormatOptions = { day: '2-digit', month: '2-digit' };

/** « 09:30 » in the local (or given) timezone. */
export function formatLocalHm(d: Date, timeZone?: string): string {
  return d.toLocaleTimeString('fr-FR', { ...HM_OPTS, timeZone });
}

/** « 08/07 à 09:30 » in the local (or given) timezone. */
export function formatLocalDayHm(d: Date, timeZone?: string): string {
  const day = d.toLocaleDateString('fr-FR', { ...DAY_OPTS, timeZone });
  return `${day} à ${formatLocalHm(d, timeZone)}`;
}

/**
 * A short, stable timezone label built from the UTC offset, e.g. « UTC−4 ».
 * Offset-based (not an abbreviation) so it is unambiguous and locale-independent.
 * `offsetMinutes` is injectable for tests; defaults to the browser's.
 */
export function utcOffsetLabel(offsetMinutes?: number): string {
  const off = offsetMinutes ?? -new Date().getTimezoneOffset(); // minutes east of UTC
  if (off === 0) return 'UTC';
  const sign = off > 0 ? '+' : '−'; // real minus sign for a clean look
  const abs = Math.abs(off);
  const h = Math.floor(abs / 60);
  const m = abs % 60;
  return `UTC${sign}${h}${m ? `:${String(m).padStart(2, '0')}` : ''}`;
}

/** « Heure locale · UTC−4 » — the discreet indicator shown near the chart. */
export function localTimeLabel(offsetMinutes?: number): string {
  return `Heure locale · ${utcOffsetLabel(offsetMinutes)}`;
}
