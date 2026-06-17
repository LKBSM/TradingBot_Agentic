/**
 * Pure builder for the chart's BOS / CHOCH break MARKERS.
 *
 * Turns the descriptive structure-event lists the backend already emits
 * (`structure.bos_events` / `structure.choch_events`) into time-anchored
 * marker descriptors. NOTHING here detects, recomputes, or projects — it only
 * reads engine-emitted events (direction + honest `broken_at`) and decides
 * where to drop an arrow. Split out of `ReadingChart` so the dedup/sort logic
 * is unit-testable without a canvas / lightweight-charts instance.
 */
import type { SeriesMarker, UTCTimestamp } from 'lightweight-charts';
import type { MarketReadingStructure } from '@/types/market-reading';

/** Marker palette — mirrors the break-level line colours in ReadingChart. */
const MARKER_COLOR = { bos: '#8B95A7', choch: '#8E84B0' } as const;

/** ISO-8601 → UNIX seconds; NaN when unparseable. */
function isoToSec(iso: string | null | undefined): number {
  if (!iso) return NaN;
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? NaN : Math.floor(ms / 1000);
}

/**
 * Build the sorted marker list for the BOS/CHOCH break history.
 *
 * Rules (descriptive only):
 *   · one arrow per event at its break bar — bullish ↑ below the bar, bearish ↓
 *     above the bar;
 *   · CHOCH wins a shared bar — a CHOCH is a reversal BOS on the SAME bar, so we
 *     drop the duplicate BOS marker at a timestamp that already has a CHOCH;
 *   · events with an unparseable timestamp are skipped;
 *   · output is sorted ascending by time (lightweight-charts requires it).
 */
export function buildStructureMarkers(
  structure: MarketReadingStructure,
): SeriesMarker<UTCTimestamp>[] {
  const chochTimes = new Set<number>();
  const markers: SeriesMarker<UTCTimestamp>[] = [];

  for (const e of structure.choch_events ?? []) {
    const t = isoToSec(e.broken_at);
    if (!Number.isFinite(t)) continue;
    chochTimes.add(t);
    const up = e.direction === 'bullish';
    markers.push({
      time: t as UTCTimestamp,
      position: up ? 'belowBar' : 'aboveBar',
      color: MARKER_COLOR.choch,
      shape: up ? 'arrowUp' : 'arrowDown',
      text: 'CHOCH',
    });
  }

  for (const e of structure.bos_events ?? []) {
    const t = isoToSec(e.broken_at);
    if (!Number.isFinite(t)) continue;
    if (chochTimes.has(t)) continue; // CHOCH already marks this bar
    const up = e.direction === 'bullish';
    markers.push({
      time: t as UTCTimestamp,
      position: up ? 'belowBar' : 'aboveBar',
      color: MARKER_COLOR.bos,
      shape: up ? 'arrowUp' : 'arrowDown',
      text: 'BOS',
    });
  }

  markers.sort((a, b) => (a.time as number) - (b.time as number));
  return markers;
}
