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
import type {
  BOSRecent,
  CHOCHRecent,
  MarketReadingStructure,
} from '@/types/market-reading';

/** Marker palette — mirrors the break-level line colours in ReadingChart. */
const MARKER_COLOR = { bos: '#8B95A7', choch: '#8E84B0' } as const;
/** VZ-1 — the SELECTED event's arrow is repainted in the accent so it stands out
 *  from the descriptive grey/violet history. */
const SELECTED_MARKER_COLOR = '#4d9de0';

/** Which event is currently selected, for emphasis — anchored by STABLE ID. */
export interface SelectedEventMarker {
  /** STR-2 defect C: the selected event's stable id (see {@link eventId}). */
  id: string;
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
 * STR-2 defect C: the STABLE id used to anchor focus on a break event. Prefers
 * the backend-emitted `event.id` (`<kind>_<iso>_<dir>`); only when a legacy
 * payload omits it do we synthesise a key. Because the id embeds the KIND, a BOS
 * and a CHOCH that share a bar have DIFFERENT ids — clicking one can never select
 * the other. The click handler, the marker emphasis and {@link findEventById} all
 * derive the id here, so they always agree.
 */
export function eventId(
  kind: 'bos' | 'choch',
  e: { id?: string | null; broken_at: string; level: number },
): string {
  return e.id ?? `${kind}:${isoToSec(e.broken_at)}:${e.level}`;
}

/**
 * Resolve a focus id back to its event, or `null` when the id matches NOTHING in
 * the current reading. STR-2 defect C: an unknown/stale id is REJECTED here — the
 * caller must not fall back to « the nearest timestamp ». Searches the journals
 * and the point-in-time fields (either can be the displayed break).
 */
export function findEventById(
  structure: MarketReadingStructure,
  id: string,
): { kind: 'bos' | 'choch'; event: BOSRecent | CHOCHRecent } | null {
  for (const e of structure.bos_events ?? []) {
    if (eventId('bos', e) === id) return { kind: 'bos', event: e };
  }
  for (const e of structure.choch_events ?? []) {
    if (eventId('choch', e) === id) return { kind: 'choch', event: e };
  }
  if (structure.current_bos && eventId('bos', structure.current_bos) === id) {
    return { kind: 'bos', event: structure.current_bos };
  }
  if (structure.current_choch && eventId('choch', structure.current_choch) === id) {
    return { kind: 'choch', event: structure.current_choch };
  }
  return null;
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
  // STR-2 defect C: emphasis is decided by STABLE ID, never by (kind, time). A
  // BOS and a CHOCH on the same bar have different ids, so the accent lands on
  // exactly the event the user clicked.
  const isSelected = (kind: 'bos' | 'choch', e: BOSRecent | CHOCHRecent) =>
    selected != null && eventId(kind, e) === selected.id;

  const chochTimes = new Set<number>();
  const markers: SeriesMarker<UTCTimestamp>[] = [];
  const inRange = (t: number) => minTime === undefined || t >= minTime;

  for (const e of structure.choch_events ?? []) {
    const t = isoToSec(e.broken_at);
    if (!Number.isFinite(t) || !inRange(t)) continue;
    chochTimes.add(t);
    const sel = isSelected('choch', e);
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
    const sel = isSelected('bos', e);
    // « CHOCH wins a shared bar » — EXCEPT when this exact BOS is the selected
    // event: then the user asked for it, so it must show (defect C robustness on
    // a shared bar, independent of the defect-B cascade fix).
    if (chochTimes.has(t) && !sel) continue;
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
