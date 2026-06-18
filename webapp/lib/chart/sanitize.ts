/**
 * Pure data-plausibility guards for the chart feed.
 *
 * Split out of `ReadingChart` so the "reject obvious garbage, never clamp a real
 * move" logic is unit-testable without a canvas. NOTHING here recomputes,
 * smooths, or reinterprets a value — it only DROPS a point that is structurally
 * impossible (≤0, non-finite, high<low…) or implausibly far from the recent
 * price (a feed glitch). A genuine large-but-real candle / tick passes through
 * untouched: we reject the data error, we never mask real volatility.
 */
import type { Candle } from '@/types/market-reading';

/**
 * True when a candle is structurally valid: every OHLC field finite and > 0,
 * low ≤ high, and open/close inside [low, high]. A bar failing this is a
 * corrupt feed row (0/negative/inverted), never legitimate market data.
 */
export function isValidBar(c: Candle): boolean {
  const { open, high, low, close } = c;
  if (![open, high, low, close].every((v) => Number.isFinite(v) && v > 0)) {
    return false;
  }
  if (low > high) return false;
  if (open < low || open > high) return false;
  if (close < low || close > high) return false;
  return true;
}

/**
 * Max intra-bar deviation a live tick may sit from the last closed close before
 * it's treated as a feed glitch. 0.5 = 50% — far beyond any real single-bar move
 * on the supported instruments (gold/FX/index, and even crypto on M15+), so it
 * only ever trips on fat-finger / zero / decimal-shift garbage, never on real
 * volatility.
 */
export const MAX_TICK_DEVIATION_PCT = 0.5;

/**
 * True when a live tick price is plausible to drive the forming candle: finite,
 * strictly > 0, and within `maxDevPct` of the reference (last closed close).
 * When the reference is unusable (non-finite / ≤0) we cannot judge deviation, so
 * we only enforce the finite/positive floor (don't reject a tick for a bad ref).
 */
export function isPlausibleTick(
  price: number,
  ref: number | null,
  maxDevPct: number = MAX_TICK_DEVIATION_PCT,
): boolean {
  if (!Number.isFinite(price) || price <= 0) return false;
  if (ref === null || !Number.isFinite(ref) || ref <= 0) return true;
  return Math.abs(price - ref) / ref <= maxDevPct;
}
