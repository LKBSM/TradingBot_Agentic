/**
 * RG-1d — coincidence of a CALENDAR reference level with a DETECTED liquidity
 * pocket.
 *
 * A temporal repère (day/week open, previous extreme) is computed from the
 * calendar; a liquidity pocket (BSL/SSL) is DETECTED by the engine. They are not
 * the same thing — a « haut de la veille » is NOT a BSL by definition, it is a
 * frequent COINCIDENCE of levels. This module reports only that coincidence,
 * verified against a real engine output (`structure.liquidity_pools`), never
 * assumed:
 *
 *   · a repère « coincides » with a pocket iff their published levels are within
 *     a tolerance derived from the average candle amplitude (see below);
 *   · the side (`bsl`/`ssl`) is taken from the MATCHED pocket, never inferred
 *     from the repère;
 *   · no match ⇒ null ⇒ the caller keeps the plain repère label.
 *
 * It states a coincidence of levels — nothing about importance, priority, a
 * target, or « the most interesting » liquidity. There is no ranking here.
 */
import type { LiquidityPool, LiquiditySide } from '@/types/market-reading';

/**
 * Tolerance (price units) for « same level »: a fraction of the recent average
 * candle amplitude (`volatility_detail.recent_avg` — the mean True Range over the
 * recent window). Retenu : **0,25 × recent_avg** (a quarter of the average
 * candle). Rationale (documented in the RG-1d audit): below a quarter-candle two
 * horizontal levels render as ONE line and describe the same price; beyond it the
 * eye separates them. Returns null when no amplitude basis is available — the
 * caller then claims NO coincidence rather than inventing one.
 */
export const COINCIDENCE_FRACTION = 0.25;

export function coincidenceTolerance(recentAvg: number | null | undefined): number | null {
  if (recentAvg == null || !Number.isFinite(recentAvg) || recentAvg <= 0) return null;
  return COINCIDENCE_FRACTION * recentAvg;
}

/**
 * The side of the DETECTED pocket whose level is closest to `price` and within
 * `tolerance`, or null when none matches (or no tolerance basis exists). Broken
 * pockets are excluded — a broken pocket no longer rests at its level, so a
 * coincidence with it would over-state live structure. Ties break on the nearest
 * level; equal distance keeps the first (stable) pocket.
 */
export function matchLiquidity(
  price: number,
  pools: readonly LiquidityPool[] | null | undefined,
  tolerance: number | null,
): LiquiditySide | null {
  if (tolerance == null || !Number.isFinite(price)) return null;
  let best: { side: LiquiditySide; dist: number } | null = null;
  for (const p of pools ?? []) {
    if (p.status === 'broken') continue; // no longer resting there
    if (!Number.isFinite(p.level)) continue;
    const dist = Math.abs(price - p.level);
    if (dist > tolerance) continue;
    if (best == null || dist < best.dist) best = { side: p.side, dist };
  }
  return best ? best.side : null;
}
