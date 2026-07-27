/**
 * Intraday SESSION view for the Régime panel, computed live on the client from
 * the session windows the SERVER published in `market_status` (MC-1 is the
 * single source of session hours — the client holds none of its own). Given the
 * windows (America/New_York wall-clock) + the reader's clock, it derives the
 * current session, the next transition and its delay, the London/NY overlap
 * state, and the market's local time.
 *
 * Sessions are a READING convention (how traders segment the day), not a market
 * property — the Concept text says so. Everything here is descriptive.
 */
import type { MarketStatusPayload } from '@/types/market-reading';

export type SessionName = 'asia' | 'london' | 'new_york' | 'overlap' | 'outside' | 'continuous';
/** What the NEXT transition leads to (a session opening, or the weekly/session close, or reopen). */
export type TransitionLabel = 'asia' | 'london' | 'new_york' | 'close' | 'open';

export interface SessionInfo {
  current: SessionName;
  /** ET « HH:MM – HH:MM » of the active session/overlap, or null. */
  currentRange: { start: string; end: string } | null;
  /** Market local time (NY wall-clock) « HH:MM », or null. */
  localTime: string | null;
  /** Next transition: what it leads to, at what ET time, in how many minutes. */
  next: { label: TransitionLabel; at: string; inMinutes: number } | null;
  /** London/NY overlap window + its state relative to now. */
  overlap: { start: string; end: string; state: 'upcoming' | 'ongoing' | 'ended' } | null;
}

const DAY = 1440;

function toMin(hhmm: string): number {
  const [h, m] = hhmm.split(':').map(Number);
  return (h ?? 0) * 60 + (m ?? 0);
}
function toHhmm(min: number): string {
  const m = ((min % DAY) + DAY) % DAY;
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
}
/** A minute-of-day is inside [start,end): handles a window that wraps midnight. */
function inWindow(t: number, start: number, end: number): boolean {
  return start <= end ? t >= start && t < end : t >= start || t < end;
}

/** NY wall-clock of `now` as minutes-of-day + « HH:MM ». */
function nyClock(now: Date, tz: string): { min: number; hhmm: string } {
  const parts = new Intl.DateTimeFormat('en-GB', {
    timeZone: tz,
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
  }).formatToParts(now);
  const h = Number(parts.find((p) => p.type === 'hour')?.value ?? '0');
  const m = Number(parts.find((p) => p.type === 'minute')?.value ?? '0');
  return { min: h * 60 + m, hhmm: `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}` };
}

/**
 * Compute the session view, or null when the payload carries no session data.
 * Continuous markets (crypto) short-circuit to `continuous`.
 */
export function computeSession(
  payload: MarketStatusPayload | null | undefined,
  now: Date,
): SessionInfo | null {
  if (!payload) return null;
  const tz = payload.session_tz ?? 'America/New_York';

  if (payload.continuous) {
    return { current: 'continuous', currentRange: null, localTime: null, next: null, overlap: null };
  }

  const windows = payload.sessions ?? [];
  if (windows.length === 0) return null;
  const { min: nowMin, hhmm: localTime } = nyClock(now, tz);

  const byName = new Map(windows.map((s) => [s.name, { start: toMin(s.start), end: toMin(s.end) }]));
  const london = byName.get('london');
  const ny = byName.get('new_york');

  // Overlap = London ∩ New York (both non-wrapping) — derived, never stored.
  let overlapWin: { start: number; end: number } | null = null;
  if (london && ny) {
    const start = Math.max(london.start, ny.start);
    const end = Math.min(london.end, ny.end);
    if (end > start) overlapWin = { start, end };
  }

  const labelAt = (t: number): SessionName => {
    const inL = london ? inWindow(t, london.start, london.end) : false;
    const inN = ny ? inWindow(t, ny.start, ny.end) : false;
    if (inL && inN) return 'overlap';
    if (inN) return 'new_york';
    if (inL) return 'london';
    for (const s of windows) {
      if (s.name !== 'london' && s.name !== 'new_york') {
        const w = byName.get(s.name)!;
        if (inWindow(t, w.start, w.end)) return s.name as SessionName;
      }
    }
    return 'outside';
  };

  // ── Market closed (weekend / holiday / daily break / lagged): no live session.
  if (payload.state !== 'open') {
    let next: SessionInfo['next'] = null;
    if (payload.next_open_ts) {
      const openMs = new Date(payload.next_open_ts).getTime();
      const inMinutes = Math.max(0, Math.round((openMs - now.getTime()) / 60000));
      next = { label: 'open', at: nyClock(new Date(openMs), tz).hhmm, inMinutes };
    }
    return {
      current: 'outside',
      currentRange: null,
      localTime,
      next,
      overlap: overlapWin
        ? { start: toHhmm(overlapWin.start), end: toHhmm(overlapWin.end), state: 'upcoming' }
        : null,
    };
  }

  // ── Market open: current session + next transition from the windows.
  const current = labelAt(nowMin);
  let currentRange: SessionInfo['currentRange'] = null;
  if (current === 'overlap' && overlapWin) currentRange = { start: toHhmm(overlapWin.start), end: toHhmm(overlapWin.end) };
  else if (current === 'new_york' && ny) currentRange = { start: toHhmm(ny.start), end: toHhmm(ny.end) };
  else if (current === 'london' && london) currentRange = { start: toHhmm(london.start), end: toHhmm(london.end) };
  else if (current !== 'outside') {
    const w = byName.get(current);
    if (w) currentRange = { start: toHhmm(w.start), end: toHhmm(w.end) };
  }

  // Next transition = the NEAREST future boundary (smallest cyclic delay) whose
  // top label differs from the current one.
  const boundaries = Array.from(new Set(windows.flatMap((s) => [toMin(s.start), toMin(s.end)])));
  let next: SessionInfo['next'] = null;
  let bestDelta = Infinity;
  for (const b of boundaries) {
    const delta = ((b - nowMin) % DAY + DAY) % DAY;
    if (delta === 0 || delta >= bestDelta) continue;
    const lab = labelAt(b);
    if (lab === current) continue;
    const label: TransitionLabel = lab === 'outside' ? 'close' : lab === 'overlap' ? 'new_york' : (lab as TransitionLabel);
    next = { label, at: toHhmm(b), inMinutes: delta };
    bestDelta = delta;
  }

  const overlap = overlapWin
    ? {
        start: toHhmm(overlapWin.start),
        end: toHhmm(overlapWin.end),
        state: (nowMin < overlapWin.start
          ? 'upcoming'
          : nowMin < overlapWin.end
            ? 'ongoing'
            : 'ended') as 'upcoming' | 'ongoing' | 'ended',
      }
    : null;

  return { current, currentRange, localTime, next, overlap };
}

/** Format a minute delay as « 1 h 40 » / « 45 min » / « 2 j 5 h » (locale-neutral units filled by i18n). */
export function splitDelay(mins: number): { d: number; h: number; m: number } {
  const d = Math.floor(mins / DAY);
  const rem = mins % DAY;
  return { d, h: Math.floor(rem / 60), m: rem % 60 };
}

/**
 * « ven. 17:00 » — the weekly close from the server's `{weekday, time}` (Python
 * weekday, Mon=0…Sun=6), localized. Returns null when absent (continuous market).
 * The weekday label is derived via Intl from the SAME weekday the server sent —
 * no second hours source, no hard-coded string.
 */
export function formatWeeklyClose(
  wc: { weekday: number; time: string } | null | undefined,
  locale: string,
): string | null {
  if (!wc) return null;
  // 2024-01-01 (UTC) is a Monday (Python weekday 0); add the offset to land on
  // the target weekday, then format its short name.
  const ref = new Date(Date.UTC(2024, 0, 1 + (((wc.weekday % 7) + 7) % 7)));
  let wd = '';
  try {
    wd = new Intl.DateTimeFormat(locale, { weekday: 'short', timeZone: 'UTC' }).format(ref);
  } catch {
    wd = '';
  }
  return `${wd} ${wc.time}`.trim();
}
