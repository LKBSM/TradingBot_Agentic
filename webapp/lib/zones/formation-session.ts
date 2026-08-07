/**
 * VZ-1 — the trading session a zone FORMED in (« Formée en session de … »).
 *
 * A reading convention, not a market property: traders segment the global day
 * into Asia / London / New York. The canonical hours live in the backend
 * (`src/intelligence/market_calendar.py::_STANDARD_SESSIONS`, America/New_York
 * wall-clock). This is a faithful client mirror so the session label needs no
 * extra request; it derives the session purely from the formation timestamp.
 *
 * Continuous markets (crypto) have no session découpage → null.
 */

import { isTwentyFourSevenMarket } from '@/lib/market-reading/session';

export type FormationSession = 'asia' | 'london' | 'new_york';

/** America/New_York wall-clock windows (minutes since midnight). Asia wraps. */
const WINDOWS: Record<FormationSession, { start: number; end: number }> = {
  asia: { start: 19 * 60, end: 4 * 60 }, // 19:00 → 04:00 (wraps past midnight)
  london: { start: 3 * 60, end: 11 * 60 + 30 }, // 03:00 → 11:30
  new_york: { start: 8 * 60, end: 17 * 60 }, // 08:00 → 17:00
};

function inWindow(minutes: number, w: { start: number; end: number }): boolean {
  return w.start <= w.end
    ? minutes >= w.start && minutes < w.end
    : minutes >= w.start || minutes < w.end; // wraps midnight
}

/** Minutes-since-midnight of `iso` in America/New_York, or null if unparsable. */
function nyMinutes(iso: string): number | null {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
  }).formatToParts(d);
  const h = Number(parts.find((p) => p.type === 'hour')?.value);
  const m = Number(parts.find((p) => p.type === 'minute')?.value);
  if (!Number.isFinite(h) || !Number.isFinite(m)) return null;
  return (h % 24) * 60 + m;
}

/**
 * The session `iso` falls in for `instrument`. Precedence New York → London →
 * Asia so the 08:00–11:30 London/NY overlap reads as New York (the anchor
 * session), matching the backend convention. null for crypto or off-session
 * formation (honest — no fabricated label).
 */
export function formationSession(
  iso: string,
  instrument: string,
): FormationSession | null {
  if (isTwentyFourSevenMarket(instrument)) return null;
  const minutes = nyMinutes(iso);
  if (minutes == null) return null;
  if (inWindow(minutes, WINDOWS.new_york)) return 'new_york';
  if (inWindow(minutes, WINDOWS.london)) return 'london';
  if (inWindow(minutes, WINDOWS.asia)) return 'asia';
  return null;
}
