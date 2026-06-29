/**
 * Pure, present-tense helpers for the enriched « Régime de marché » section.
 *
 * Every function here only READS facts the engine already published in the
 * MarketReading (structure + header) and presents them descriptively. They never
 * detect, recompute, score or predict. A derived count (bars since a break, the
 * number of active zones) is plain arithmetic over engine-emitted timestamps and
 * lifecycle statuses — never a detection threshold. When the underlying fact is
 * absent, the helper returns `null` so the caller renders « non disponible »
 * instead of inventing a value.
 */
import type {
  BOSRecent,
  CHOCHRecent,
  MarketReadingHeader,
  MarketReadingStructure,
  ValidationStatus,
} from '@/types/market-reading';

// ─── Timeframe → minutes (for the « X bougies » derivation) ──────────────────

const TIMEFRAME_MINUTES: Record<string, number> = {
  M1: 1,
  M5: 5,
  M15: 15,
  M30: 30,
  H1: 60,
  H4: 240,
  D1: 1_440,
  W1: 10_080,
};

/** Minutes per candle for a timeframe code, or null when the code is unknown. */
export function timeframeMinutes(timeframe: string): number | null {
  return TIMEFRAME_MINUTES[timeframe] ?? null;
}

// ─── Break timestamp (engine wall-clock, no timezone math) ────────────────────

/**
 * Render an ISO timestamp as « JJ/MM à HH:MM », reading the calendar/clock
 * fields AS AUTHORED by the engine (no UTC conversion). This keeps the displayed
 * moment identical to what the engine emitted and makes the output deterministic
 * regardless of the runtime timezone. Returns null on an unparseable string.
 */
export function formatBreakTimestamp(iso: string): string | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})/.exec(iso);
  if (!m) return null;
  const [, , month, day, hh, mi] = m;
  return `${day}/${month} à ${hh}:${mi}`;
}

// ─── (b) Trend maturity — anchored on the last CHOCH ─────────────────────────

export interface TrendMaturity {
  /** Orientation established by the last CHOCH (a structural fact). */
  direction: 'bullish' | 'bearish';
  /** ISO timestamp of that CHOCH break. */
  brokenAt: string;
  /**
   * Whole candles between the break and the reading's candle close, or null when
   * not derivable (unknown timeframe, unparseable / future timestamp). Derived,
   * never a detection input.
   */
  bars: number | null;
}

/**
 * Maturity of the current structural orientation, taken from the last CHOCH the
 * engine surfaces (`structure.choch`). Returns null when no CHOCH is available —
 * the caller then shows « non disponible ». Read-only/descriptive.
 */
export function deriveTrendMaturity(
  structure: MarketReadingStructure,
  header: MarketReadingHeader,
): TrendMaturity | null {
  const choch = structure.choch;
  if (!choch) return null;

  let bars: number | null = null;
  const tfMin = timeframeMinutes(header.timeframe);
  if (tfMin) {
    const closeMs = new Date(header.candle_close_ts).getTime();
    const breakMs = new Date(choch.broken_at).getTime();
    const diffMs = closeMs - breakMs;
    // Guard against unparseable dates and the known « broken_at in the future »
    // data glitch — we never show a negative or NaN candle count.
    if (Number.isFinite(diffMs) && diffMs >= 0) {
      bars = Math.floor(diffMs / (tfMin * 60_000));
    }
  }

  return { direction: choch.direction, brokenAt: choch.broken_at, bars };
}

/**
 * Present-tense maturity line, e.g.
 *   « Structure orientée haussière depuis le CHOCH du 24/06 à 14:30 (≈ 18 bougies M15). »
 * Returns null when no CHOCH is available (caller → « non disponible »).
 */
export function formatTrendMaturity(
  structure: MarketReadingStructure,
  header: MarketReadingHeader,
): string | null {
  const m = deriveTrendMaturity(structure, header);
  if (!m) return null;

  const orient = m.direction === 'bullish' ? 'haussière' : 'baissière';
  const when = formatBreakTimestamp(m.brokenAt);
  const whenPart = when ? ` du ${when}` : '';
  const barsPart =
    m.bars != null
      ? ` (≈ ${m.bars} bougie${m.bars > 1 ? 's' : ''} ${header.timeframe})`
      : '';
  return `Structure orientée ${orient} depuis le CHOCH${whenPart}${barsPart}.`;
}

// ─── (c) Last structural event — most recent of BOS / CHOCH ──────────────────

const VALIDATION_MASC: Record<ValidationStatus, string> = {
  confirmed: 'confirmé',
  pending: 'en attente de confirmation',
  invalidated: 'invalidé',
};

/**
 * The most recent structural break the engine surfaces, phrased as
 *   « CHOCH baissier confirmé (H1) ».
 * Picks the later of `structure.bos` / `structure.choch` by `broken_at`.
 * Returns null when neither is present (caller → « non disponible »).
 */
export function formatLastStructuralEvent(
  structure: MarketReadingStructure,
  header: MarketReadingHeader,
): string | null {
  const { bos, choch } = structure;

  let kind: 'BOS' | 'CHOCH' | null = null;
  let event: BOSRecent | CHOCHRecent | null = null;
  if (bos && choch) {
    const newerIsBos =
      new Date(bos.broken_at).getTime() >= new Date(choch.broken_at).getTime();
    kind = newerIsBos ? 'BOS' : 'CHOCH';
    event = newerIsBos ? bos : choch;
  } else if (bos) {
    kind = 'BOS';
    event = bos;
  } else if (choch) {
    kind = 'CHOCH';
    event = choch;
  }
  if (!kind || !event) return null;

  const dir = event.direction === 'bullish' ? 'haussier' : 'baissier';
  const val = VALIDATION_MASC[event.validation_status];
  return `${kind} ${dir} ${val} (${header.timeframe})`;
}

// ─── (d) Active zone density ──────────────────────────────────────────────────

export interface ZoneDensity {
  /** Active Order Blocks (status === 'active'). */
  ob: number;
  /** Active Fair Value Gaps (status === 'active'). */
  fvg: number;
}

/**
 * Count the currently-active OB / FVG zones. Always available (the lists default
 * to empty), so 0 · 0 is a real fact — never « non disponible ».
 */
export function countActiveZones(
  structure: MarketReadingStructure,
): ZoneDensity {
  const ob = structure.order_blocks.filter((z) => z.status === 'active').length;
  const fvg = structure.fair_value_gaps.filter(
    (z) => z.status === 'active',
  ).length;
  return { ob, fvg };
}

/** « 3 OB · 5 FVG actifs ». */
export function formatZoneDensity(structure: MarketReadingStructure): string {
  const { ob, fvg } = countActiveZones(structure);
  return `${ob} OB · ${fvg} FVG actifs`;
}
