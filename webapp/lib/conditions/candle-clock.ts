/**
 * Candle-clock helpers for the Scanner — pure, side-effect-free, UTC-aligned.
 *
 * The engine's structural readings only change when a candle CLOSES (SMC law,
 * respected throughout). So the honest cadence for an auto-refresh is "at the
 * next candle close", per timeframe — never a per-second poll. These functions
 * compute that boundary and render an honest freshness label. No prediction,
 * no live-tick path: we read what is already produced.
 */

/** Minutes per timeframe. Mirrors the backend `_TIMEFRAME_MINUTES`. */
const TIMEFRAME_MINUTES: Record<string, number> = {
  M1: 1,
  M5: 5,
  M15: 15,
  M30: 30,
  H1: 60,
  H4: 240,
  D1: 1440,
  W1: 10080,
};

/** Minutes for a timeframe code, or null if unknown. */
export function timeframeToMinutes(timeframe: string): number | null {
  return TIMEFRAME_MINUTES[timeframe.toUpperCase()] ?? null;
}

/**
 * The next candle-close boundary (epoch ms) at/after `nowMs` for one timeframe.
 *
 * Intraday and daily boundaries align to the UTC epoch, which lands them on the
 * conventional marks: M15 → :00/:15/:30/:45, H1 → top of hour, H4 → 00/04/.../20
 * UTC, D1 → 00:00 UTC. Unknown/weekly timeframes are not used by the scanner;
 * they fall back to a 15-minute cadence so a caller never hangs without a tick.
 */
function nextBoundaryForTimeframe(timeframe: string, nowMs: number): number {
  const minutes = timeframeToMinutes(timeframe);
  const ONE_DAY_MIN = 1440; // epoch-aligned ceiling (D1); W1+ is broker-anchored.
  // W1 (and anything unknown) is broker-anchored, not epoch-aligned — don't
  // pretend otherwise; degrade to a 15-minute tick rather than mislead.
  const stepMin = minutes && minutes <= ONE_DAY_MIN ? minutes : 15;
  const stepMs = stepMin * 60_000;
  // ceil to the next strictly-future multiple of stepMs from the epoch.
  return (Math.floor(nowMs / stepMs) + 1) * stepMs;
}

/**
 * The SOONEST next candle close (epoch ms) across the given timeframes — i.e.
 * the moment at which at least one of the scanned readings can have changed.
 * Returns null for an empty set.
 */
export function nextCandleCloseMs(timeframes: string[], nowMs: number): number | null {
  if (timeframes.length === 0) return null;
  let soonest = Infinity;
  for (const tf of timeframes) {
    const b = nextBoundaryForTimeframe(tf, nowMs);
    if (b < soonest) soonest = b;
  }
  return Number.isFinite(soonest) ? soonest : null;
}

/**
 * Honest "last analysis" label from an ISO timestamp (e.g. the scan `as_of`).
 * "à l'instant" under a minute, then minutes / hours / days. Returns null for a
 * missing or unparseable timestamp rather than inventing freshness.
 */
export function freshnessLabel(iso: string | null | undefined, nowMs: number): string | null {
  if (!iso) return null;
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return null;
  const seconds = Math.max(0, Math.round((nowMs - then) / 1000));
  if (seconds < 60) return "à l'instant";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `il y a ${minutes} min`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `il y a ${hours} h`;
  const days = Math.round(hours / 24);
  return `il y a ${days} j`;
}
