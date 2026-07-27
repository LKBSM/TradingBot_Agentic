/**
 * Pure helpers for the enriched « Régime de marché » panel — the two measures
 * that are ARITHMETIC over engine-emitted candles/structure, not new detections:
 *
 *   · Position dans le range (measure 4) — where the price sits between the
 *     engine's structural range extremes.
 *   · Niveaux de référence (measure 10) — day/week open + previous day/week
 *     extremes, read straight off the DATA FEED's own D1 / W1 candles.
 *
 * The « day » / « week » boundary is therefore the feed's own D1 / W1 candle
 * (the same reference the chart uses) — a single definition, never a second
 * client-side boundary. See docs/audits/AUDIT-rg-1-regime-enrichi.md.
 *
 * Every function returns `null` for a value it cannot compute (missing candles,
 * a degenerate range) so the caller renders nothing rather than inventing a
 * number — « pas de donnée, pas de ligne ».
 */
import type { Candle, MarketReadingStructure } from '@/types/market-reading';

// ─── Position dans le range (measure 4) ──────────────────────────────────────

export interface StructureRange {
  /** Lower structural bound (engine range extreme). */
  low: number;
  /** Upper structural bound (engine range extreme). */
  high: number;
}

/**
 * The structural range bounds, taken from the engine's liquidity pools —
 * `kind: "range_low"` / `"range_high"` are the window's extreme swings the SMC
 * mapper emits. These are DESCRIPTIVE window extremes (max/min of the retained
 * swings), NOT « the last swing » — the panel labels them as such. Returns null
 * when either bound is missing or the range is degenerate (high <= low).
 */
export function structureRange(
  structure: Pick<MarketReadingStructure, 'liquidity_pools'>,
): StructureRange | null {
  const pools = structure.liquidity_pools ?? [];
  let high: number | null = null;
  let low: number | null = null;
  for (const p of pools) {
    if (p.kind === 'range_high') high = p.level;
    else if (p.kind === 'range_low') low = p.level;
  }
  if (high == null || low == null || high <= low) return null;
  return { low, high };
}

/**
 * Where `price` sits between `low` and `high`, as a 0–100 %. 0 % = lower bound,
 * 100 % = upper bound. This is plain arithmetic over the engine's own bounds
 * (the engine does not itself compute a « position »), clamped to [0, 100] so a
 * price temporarily outside the retained range still reads as an edge rather
 * than an absurd value. Returns null on a degenerate range.
 */
export function positionPct(low: number, high: number, price: number): number | null {
  if (!(high > low)) return null;
  const pct = ((price - low) / (high - low)) * 100;
  return Math.max(0, Math.min(100, pct));
}

// ─── Niveaux de référence (measure 10) ───────────────────────────────────────

export interface ReferenceLevels {
  /** Open of the current (forming) daily candle. */
  dayOpen: number | null;
  /** Open of the current (forming) weekly candle. */
  weekOpen: number | null;
  /** High / low of the PREVIOUS completed daily candle. */
  prevDayHigh: number | null;
  prevDayLow: number | null;
  /** High / low of the PREVIOUS completed weekly candle. */
  prevWeekHigh: number | null;
  prevWeekLow: number | null;
}

/** Open of the most recent candle (the current, still-forming period). */
function currentOpen(candles: Candle[]): number | null {
  const last = candles.at(-1);
  return last ? last.open : null;
}

/** High/low of the previous completed candle (second from the end). */
function previousExtremes(candles: Candle[]): { high: number | null; low: number | null } {
  // Need at least [previous, current]; the last item is the forming period.
  if (candles.length < 2) return { high: null, low: null };
  const prev = candles[candles.length - 2]!;
  return { high: prev.high, low: prev.low };
}

/**
 * Assemble the calendar reference levels from the feed's D1 and W1 candle series
 * (ascending, oldest-first — as `/api/candles` returns them). Any series that is
 * empty/too short yields nulls for its levels; the caller drops those lines.
 */
export function referenceLevels(
  daily: Candle[],
  weekly: Candle[],
): ReferenceLevels {
  const day = previousExtremes(daily);
  const week = previousExtremes(weekly);
  return {
    dayOpen: currentOpen(daily),
    weekOpen: currentOpen(weekly),
    prevDayHigh: day.high,
    prevDayLow: day.low,
    prevWeekHigh: week.high,
    prevWeekLow: week.low,
  };
}

/** Signed % distance from `price` to `level` ((level − price) / price). */
export function distancePct(level: number, price: number): number | null {
  if (!(price > 0) || !Number.isFinite(level)) return null;
  return ((level - price) / price) * 100;
}
