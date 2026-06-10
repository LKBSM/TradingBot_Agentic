import { describe, expect, it } from 'vitest';
import { computeDailyChange } from './price';
import type { Candle } from '@/types/market-reading';

/** Build a candle whose close (and time) are all we care about here. */
function candle(iso: string, close: number): Candle {
  const time = Math.floor(Date.UTC(...isoParts(iso)) / 1000);
  return { time, open: close, high: close, low: close, close };
}

/** Parse "YYYY-MM-DDTHH:mm" into Date.UTC args. */
function isoParts(
  iso: string,
): [number, number, number, number, number] {
  const m = iso.match(/^(\d+)-(\d+)-(\d+)T(\d+):(\d+)/)!;
  return [
    Number(m[1]),
    Number(m[2]) - 1,
    Number(m[3]),
    Number(m[4]),
    Number(m[5]),
  ];
}

describe('computeDailyChange (descriptive daily change)', () => {
  it('returns null for an empty / missing window', () => {
    expect(computeDailyChange([])).toBeNull();
    expect(computeDailyChange(null)).toBeNull();
    expect(computeDailyChange(undefined)).toBeNull();
  });

  it('uses the last candle close as the unified price', () => {
    const out = computeDailyChange([
      candle('2026-05-25T23:45', 100),
      candle('2026-05-26T00:15', 105.5),
    ]);
    expect(out?.price).toBe(105.5);
  });

  it('references the previous UTC day last close for the change', () => {
    const out = computeDailyChange([
      candle('2026-05-25T22:00', 100),
      candle('2026-05-25T23:00', 101),
      candle('2026-05-25T23:45', 102), // previous-day last close → reference
      candle('2026-05-26T00:00', 103),
      candle('2026-05-26T00:15', 105), // latest → price
    ]);
    expect(out?.referenceClose).toBe(102);
    expect(out?.price).toBe(105);
    expect(out?.changeAbs).toBeCloseTo(3, 10);
    expect(out?.changePct).toBeCloseTo(3 / 102, 10);
  });

  it('skips weekends — previous *populated* day, not previous calendar day', () => {
    // Friday close then Monday: reference should be Friday's last close.
    const out = computeDailyChange([
      candle('2026-05-22T20:45', 200), // Friday
      candle('2026-05-25T00:00', 204), // Monday
      candle('2026-05-25T00:15', 210),
    ]);
    expect(out?.referenceClose).toBe(200);
    expect(out?.changePct).toBeCloseTo((210 - 200) / 200, 10);
  });

  it('yields a null change when the window has only one UTC day', () => {
    const out = computeDailyChange([
      candle('2026-05-26T00:00', 100),
      candle('2026-05-26T00:15', 101),
    ]);
    expect(out?.price).toBe(101);
    expect(out?.referenceClose).toBeNull();
    expect(out?.changePct).toBeNull();
    expect(out?.changeAbs).toBeNull();
  });

  it('produces a negative change when price fell vs the reference', () => {
    const out = computeDailyChange([
      candle('2026-05-25T23:45', 4271.0), // reference
      candle('2026-05-26T00:15', 4131.4), // latest (founder example order of magnitude)
    ]);
    expect(out?.changePct).toBeLessThan(0);
  });
});
