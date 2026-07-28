/**
 * Pure helpers for the enriched « Régime de marché » panel — the two measures
 * that are ARITHMETIC over engine output, not new detections:
 *
 *   · Position dans le range (measure 4) — where the price sits between the
 *     engine's structural range extremes.
 *   · Niveaux de référence (measure 10) — day/week open + previous day/week
 *     extremes. These are computed SERVER-SIDE (RG-1c): the backend aggregates
 *     max(high)/min(low) over every candle of each MC-1 trading day/week, in the
 *     instrument's NY wall-clock — ONE definition of « a trading day », per
 *     instrument. The front only maps the server payload here; it never indexes a
 *     single candle (reading one bar's high/low was the RG-1c bug: a sub-daily
 *     range presented as a full day). See docs/audits/AUDIT-rg-1c-niveaux-correctif.md.
 *
 * Every function returns `null` for a value it cannot compute (missing data, a
 * degenerate range, an incompletely-covered period) so the caller renders
 * nothing rather than inventing a number — « pas de donnée, pas de ligne ».
 */
import type {
  MarketReadingStructure,
  ReferenceLevelsPayload,
} from '@/types/market-reading';

// ─── Position dans le range (measure 4) ──────────────────────────────────────

export interface StructureRange {
  /** Lower structural bound (engine range extreme). */
  low: number;
  /** Upper structural bound (engine range extreme). */
  high: number;
}

/**
 * The structural range bounds, taken from the engine's liquidity pools. These
 * are DESCRIPTIVE structural extremes (from the retained swings), NOT raw candle
 * highs/lows — the panel labels them as such.
 *
 * `kind: "range_high"`/`"range_low"` are the direct extremes, BUT the mapper
 * omits them when an equal-highs/-lows cluster already sits AT that extreme (it
 * emits the cluster pool instead). So we take the top bound as the highest
 * `is_external` BSL pool level (range_high OR the equal-highs cluster at the top)
 * and the bottom bound as the lowest `is_external` SSL pool level — robust
 * whenever structure exists. Returns null when either side is missing or the
 * range is degenerate (high <= low).
 */
export function structureRange(
  structure: Pick<MarketReadingStructure, 'liquidity_pools'>,
): StructureRange | null {
  const pools = structure.liquidity_pools ?? [];
  let high: number | null = null;
  let low: number | null = null;
  for (const p of pools) {
    // Direct range extremes win outright.
    if (p.kind === 'range_high') high = p.level;
    else if (p.kind === 'range_low') low = p.level;
  }
  // Fallback to the external clusters at the top / bottom of the window.
  if (high == null) {
    for (const p of pools) {
      if (p.is_external && p.side === 'bsl' && (high == null || p.level > high)) high = p.level;
    }
  }
  if (low == null) {
    for (const p of pools) {
      if (p.is_external && p.side === 'ssl' && (low == null || p.level < low)) low = p.level;
    }
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
  /** Open of the current (forming) trading day. */
  dayOpen: number | null;
  /** Open of the current (forming) trading week. */
  weekOpen: number | null;
  /** High / low of the PREVIOUS complete trading day. */
  prevDayHigh: number | null;
  prevDayLow: number | null;
  /** High / low of the PREVIOUS complete trading week. */
  prevWeekHigh: number | null;
  prevWeekLow: number | null;
  /** Whether the previous-day window was fully covered by the cached candles. */
  dayComplete: boolean;
  /** Whether the previous-week window was fully covered by the cached candles. */
  weekComplete: boolean;
}

/**
 * Map the SERVER-aggregated reference levels (RG-1c) to the panel's shape. The
 * backend has already aggregated max/min over every candle of each MC-1 trading
 * period and dropped any period it could not fully cover (those arrive as null).
 * Returns null when the payload is absent (static fixture / older backend), so
 * the caller drops the tile rather than inventing values.
 */
export function referenceLevelsFromPayload(
  payload: ReferenceLevelsPayload | null | undefined,
): ReferenceLevels | null {
  if (!payload) return null;
  return {
    dayOpen: payload.day_open,
    weekOpen: payload.week_open,
    prevDayHigh: payload.prev_day_high,
    prevDayLow: payload.prev_day_low,
    prevWeekHigh: payload.prev_week_high,
    prevWeekLow: payload.prev_week_low,
    dayComplete: payload.day_complete,
    weekComplete: payload.week_complete,
  };
}

/** Signed % distance from `price` to `level` ((level − price) / price). */
export function distancePct(level: number, price: number): number | null {
  if (!(price > 0) || !Number.isFinite(level)) return null;
  return ((level - price) / price) * 100;
}
