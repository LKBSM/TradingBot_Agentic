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
/** VZ-1 — the SELECTED event's arrow is repainted in the accent so it stands out
 *  from the descriptive grey/violet history. */
const SELECTED_MARKER_COLOR = '#4d9de0';

/** Which event (kind + confirmation time) is currently selected, for emphasis. */
export interface SelectedEventMarker {
  kind: 'bos' | 'choch';
  /** Confirmation candle time, epoch seconds. */
  atSec: number;
}

export interface StructureMarkerOptions {
  /** The selected event to emphasise (accent colour), or null. */
  selected?: SelectedEventMarker | null;
  /** When true, return ONLY the selected event's marker (breaks layer hidden). */
  onlySelected?: boolean;
}

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
 *   · events OLDER than `minTime` are dropped — the backend collects events over
 *     its full 500-bar window while the chart loads fewer candles, and
 *     lightweight-charts v5 CLAMPS out-of-range markers onto the first loaded
 *     bar (NearestRight in createSeriesMarkers) instead of ignoring them, which
 *     stacked stale labels vertically at the left edge;
 *   · output is sorted ascending by time (lightweight-charts requires it).
 *
 * @param minTime UNIX seconds of the first loaded candle; events breaking
 *   before it have no bar to anchor to and are omitted. Omit to keep all.
 */
export function buildStructureMarkers(
  structure: MarketReadingStructure,
  minTime?: number,
  options?: StructureMarkerOptions,
): SeriesMarker<UTCTimestamp>[] {
  const selected = options?.selected ?? null;
  const onlySelected = options?.onlySelected ?? false;
  const isSelected = (kind: 'bos' | 'choch', t: number) =>
    selected != null && selected.kind === kind && selected.atSec === t;

  const chochTimes = new Set<number>();
  const markers: SeriesMarker<UTCTimestamp>[] = [];
  const inRange = (t: number) => minTime === undefined || t >= minTime;

  for (const e of structure.choch_events ?? []) {
    const t = isoToSec(e.broken_at);
    if (!Number.isFinite(t) || !inRange(t)) continue;
    chochTimes.add(t);
    const sel = isSelected('choch', t);
    if (onlySelected && !sel) continue;
    const up = e.direction === 'bullish';
    markers.push({
      time: t as UTCTimestamp,
      position: up ? 'belowBar' : 'aboveBar',
      color: sel ? SELECTED_MARKER_COLOR : MARKER_COLOR.choch,
      shape: up ? 'arrowUp' : 'arrowDown',
      text: 'CHOCH',
    });
  }

  for (const e of structure.bos_events ?? []) {
    const t = isoToSec(e.broken_at);
    if (!Number.isFinite(t) || !inRange(t)) continue;
    if (chochTimes.has(t)) continue; // CHOCH already marks this bar
    const sel = isSelected('bos', t);
    if (onlySelected && !sel) continue;
    const up = e.direction === 'bullish';
    markers.push({
      time: t as UTCTimestamp,
      position: up ? 'belowBar' : 'aboveBar',
      color: sel ? SELECTED_MARKER_COLOR : MARKER_COLOR.bos,
      shape: up ? 'arrowUp' : 'arrowDown',
      text: 'BOS',
    });
  }

  markers.sort((a, b) => (a.time as number) - (b.time as number));
  return markers;
}
