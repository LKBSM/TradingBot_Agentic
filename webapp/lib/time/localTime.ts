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

// 24-hour clock in EVERY locale (`hour12: false`): a trading axis reads « 14:30 »,
// never « 02:30 PM ». en-US would otherwise default to 12-hour AM/PM, breaking
// both consistency with fr/es and the compact axis width.
const HM_OPTS: Intl.DateTimeFormatOptions = { hour: '2-digit', minute: '2-digit', hour12: false };
// Day + short MONTH NAME (not a numeric month): the named month is unambiguous
// across languages, where a numeric « 01/09 » reads as 1 Sep in fr/es but as
// Jan 9 in en. The locale also drives the day/month ORDER (fr « 31 août », en
// « Aug 31 »). I18N-1: every axis/annotation date uses this — never day/month
// digits — so an English reader never mistakes the day for the month.
const DAY_NAMED_OPTS: Intl.DateTimeFormatOptions = { day: 'numeric', month: 'short' };

/** « 09:30 » in the given locale + (optional) timezone. */
export function formatLocalHm(d: Date, locale: string, timeZone?: string): string {
  return d.toLocaleTimeString(locale, { ...HM_OPTS, timeZone });
}

/**
 * « 31 août · 23:30 » (fr) / « Aug 31 · 23:30 » (en) / « 31 ago · 23:30 » (es) —
 * unambiguous day + named month + time in the given locale. The middot join is
 * locale-neutral punctuation (no « à »/« at » to translate). Used by the chart
 * crosshair / time-axis, where a numeric day/month would be dangerous in English.
 */
export function formatLocalDayHm(d: Date, locale: string, timeZone?: string): string {
  const day = d.toLocaleDateString(locale, { ...DAY_NAMED_OPTS, timeZone });
  return `${day} · ${formatLocalHm(d, locale, timeZone)}`;
}

/**
 * « 24 juil. à 09:45 » — LONG, localized date + time. The month is a localized
 * short name and the day/month order follows the locale (fr « 24 juil. », en
 * « Jul 24 », es « 24 jul »); the join is locale-driven (fr « à », es « a las »,
 * en « at »). Used by the Régime panel where a numeric « 24/07 » would be
 * ambiguous across languages.
 */
export function formatLocalDayLong(d: Date, locale: string, timeZone?: string): string {
  const day = d.toLocaleDateString(locale, { ...DAY_NAMED_OPTS, timeZone });
  const lc = locale.toLowerCase();
  const sep = lc.startsWith('fr') ? 'à' : lc.startsWith('es') ? 'a las' : 'at';
  return `${day} ${sep} ${formatLocalHm(d, locale, timeZone)}`;
}

/**
 * A short, stable timezone label built from the UTC offset, e.g. « UTC−4 ».
 * Offset-based (not an abbreviation) so it is unambiguous and locale-independent
 * — the SAME in every language, only the surrounding « Heure locale »/« Local
 * time »/« Hora local » prefix is translated (by the i18n caller). `offsetMinutes`
 * is injectable for tests; defaults to the browser's.
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
