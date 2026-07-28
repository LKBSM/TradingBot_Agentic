import { describe, it, expect } from 'vitest';
import {
  computeReferenceViewRange,
  REFERENCE_VIEW_MARGIN_FRAC,
} from '../referenceView';

const candles = [
  { high: 2405, low: 2395 },
  { high: 2410, low: 2398 }, // window high 2410
  { high: 2402, low: 2388 }, // window low 2388
];

describe('computeReferenceViewRange (RG-1d)', () => {
  it('spans the level AND the visible candle extremes, with a margin', () => {
    // A level far BELOW the candles (e.g. previous-week low at 2360).
    const r = computeReferenceViewRange(2360, candles, 0, 2)!;
    // Union = [2360, 2410]; pad = 50 * 0.08 = 4.
    expect(REFERENCE_VIEW_MARGIN_FRAC).toBe(0.08);
    expect(r.from).toBeCloseTo(2356, 6); // 2360 − 4
    expect(r.to).toBeCloseTo(2414, 6); // 2410 + 4
    // Both the level and the current price (a candle, within [2388, 2410]) are inside.
    expect(r.from).toBeLessThan(2360);
    expect(r.to).toBeGreaterThan(2410);
  });

  it('keeps the candles visible even when the level sits inside their band', () => {
    const r = computeReferenceViewRange(2400, candles, 0, 2)!;
    // Union stays [2388, 2410] (level already inside) — context preserved.
    expect(r.from).toBeLessThan(2388);
    expect(r.to).toBeGreaterThan(2410);
  });

  it('clamps out-of-bounds indices instead of reading past the array', () => {
    const r = computeReferenceViewRange(2360, candles, -5, 99)!;
    expect(r.from).toBeCloseTo(2356, 6);
    expect(r.to).toBeCloseTo(2414, 6);
  });

  it('pads around the level when no candle is in range (degenerate)', () => {
    const r = computeReferenceViewRange(2360, [], 0, 0)!;
    expect(r.from).toBeLessThan(2360);
    expect(r.to).toBeGreaterThan(2360);
    expect(r.to - r.from).toBeGreaterThan(0);
  });

  it('returns null for a non-finite level', () => {
    expect(computeReferenceViewRange(Number.NaN, candles, 0, 2)).toBeNull();
  });
});
