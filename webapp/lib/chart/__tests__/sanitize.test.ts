import { describe, expect, it } from 'vitest';
import { isPlausibleTick, isValidBar, MAX_TICK_DEVIATION_PCT } from '../sanitize';
import type { Candle } from '@/types/market-reading';

function bar(o: number, h: number, l: number, c: number): Candle {
  return { time: 1000, open: o, high: h, low: l, close: c };
}

describe('isValidBar', () => {
  it('accepts a normal OHLC bar', () => {
    expect(isValidBar(bar(100, 102, 99, 101))).toBe(true);
  });

  it('rejects zero / negative values', () => {
    expect(isValidBar(bar(0, 102, 99, 101))).toBe(false);
    expect(isValidBar(bar(100, 102, -1, 101))).toBe(false);
  });

  it('rejects non-finite values', () => {
    expect(isValidBar(bar(100, Number.NaN, 99, 101))).toBe(false);
    expect(isValidBar(bar(100, Infinity, 99, 101))).toBe(false);
  });

  it('rejects an inverted bar (high < low)', () => {
    expect(isValidBar(bar(100, 99, 102, 101))).toBe(false);
  });

  it('rejects an open/close outside [low, high]', () => {
    expect(isValidBar(bar(98, 102, 99, 101))).toBe(false); // open below low
    expect(isValidBar(bar(100, 102, 99, 103))).toBe(false); // close above high
  });
});

describe('isPlausibleTick', () => {
  const ref = 2000; // last closed close

  it('accepts a tick close to the reference', () => {
    expect(isPlausibleTick(2010, ref)).toBe(true);
  });

  it('accepts a genuine LARGE-but-real move (never clamps real volatility)', () => {
    // +40% is huge but under the 50% glitch threshold → passes.
    expect(isPlausibleTick(ref * (1 + MAX_TICK_DEVIATION_PCT * 0.8), ref)).toBe(true);
  });

  it('rejects zero / negative ticks', () => {
    expect(isPlausibleTick(0, ref)).toBe(false);
    expect(isPlausibleTick(-5, ref)).toBe(false);
  });

  it('rejects a tick implausibly far from the reference (feed glitch)', () => {
    expect(isPlausibleTick(ref * 5, ref)).toBe(false); // decimal-shift garbage
    expect(isPlausibleTick(ref * 0.1, ref)).toBe(false);
  });

  it('rejects non-finite ticks', () => {
    expect(isPlausibleTick(Number.NaN, ref)).toBe(false);
  });

  it('only enforces the finite/positive floor when the reference is unusable', () => {
    // No usable ref → cannot judge deviation; accept any finite positive tick.
    expect(isPlausibleTick(2000, null)).toBe(true);
    expect(isPlausibleTick(2000, 0)).toBe(true);
    expect(isPlausibleTick(0, null)).toBe(false);
  });
});
