/** LP-1 — pure chart math for the illustration candles. Deterministic; no state.
 * Candles are derived from a close series with fixed synthetic wicks so SSR and
 * client render byte-identical SVG. */
import { DEMO_CLOSES } from './data';

export interface Bounds {
  min: number;
  max: number;
}

export interface Candle {
  o: number;
  h: number;
  l: number;
  c: number;
  up: boolean;
}

/** Build OHLC candles from the close series (open = previous close, fixed wick). */
export function buildCandles(closes: readonly number[] = DEMO_CLOSES): Candle[] {
  return closes.map((c, i) => {
    const o = i === 0 ? c - 0.6 : closes[i - 1] ?? c;
    const up = c >= o;
    const body = Math.abs(c - o);
    const wick = 0.8 + (body % 1.3); // deterministic, series-derived
    const h = Math.max(o, c) + wick * 0.6;
    const l = Math.min(o, c) - wick * 0.6;
    return { o, h, l, c, up };
  });
}

/** Price bounds padded by the given extra prices (levels/labels) and a margin. */
export function priceBounds(
  candles: Candle[],
  extra: readonly number[] = [],
): Bounds {
  let min = Infinity;
  let max = -Infinity;
  for (const k of candles) {
    if (k.l < min) min = k.l;
    if (k.h > max) max = k.h;
  }
  for (const p of extra) {
    if (p < min) min = p;
    if (p > max) max = p;
  }
  const pad = (max - min) * 0.08 || 1;
  return { min: min - pad, max: max + pad };
}

export function xForIndex(i: number, n: number, w: number, padRight = 64): number {
  const usable = w - padRight - 12;
  return 12 + (usable * i) / Math.max(1, n - 1);
}

export function yForPrice(p: number, b: Bounds, h: number): number {
  const t = (p - b.min) / (b.max - b.min || 1);
  return h - t * h;
}

/** Percent helpers for absolutely-positioned overlays over the SVG box. */
export function yPct(p: number, b: Bounds): number {
  const t = (p - b.min) / (b.max - b.min || 1);
  return (1 - t) * 100;
}
