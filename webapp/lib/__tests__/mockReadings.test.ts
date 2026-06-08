import { describe, expect, it } from 'vitest';
import {
  DEFAULT_COMBO,
  getMockCandles,
  getMockReading,
} from '@/lib/mockReadings';
import { SUPPORTED_COMBOS } from '@/lib/market-reading/store';

describe('mockReadings — readings', () => {
  it('covers every V1 combo with a schema-valid reading', () => {
    for (const combo of SUPPORTED_COMBOS) {
      const reading = getMockReading(combo.instrument, combo.timeframe);
      expect(reading, `${combo.instrument}:${combo.timeframe}`).not.toBeNull();
      expect(reading!.header.instrument).toBe(combo.instrument);
      expect(reading!.header.timeframe).toBe(combo.timeframe);
      expect(reading!.schema_version).toBe('2.0.0');
    }
  });

  it('returns null for an out-of-catalogue combo', () => {
    expect(getMockReading('BTCUSD', 'M15')).toBeNull();
    expect(getMockReading('XAUUSD', 'M1')).toBeNull();
  });

  it('defaults to XAU/USD M15', () => {
    expect(DEFAULT_COMBO).toEqual({ instrument: 'XAUUSD', timeframe: 'M15' });
  });
});

describe('mockReadings — candles', () => {
  it('returns a deterministic candle series for an available combo', () => {
    const a = getMockCandles('XAUUSD', 'M15');
    const b = getMockCandles('XAUUSD', 'M15');
    expect(a).not.toBeNull();
    expect(a!.length).toBeGreaterThan(10);
    expect(b).toEqual(a); // deterministic (seeded PRNG)
  });

  it('ends exactly on the reading close price', () => {
    const reading = getMockReading('EURUSD', 'H1')!;
    const candles = getMockCandles('EURUSD', 'H1')!;
    const last = candles[candles.length - 1]!;
    expect(last.close).toBeCloseTo(reading.header.close_price, 6);
  });

  it('produces an envelope that brackets every structure level', () => {
    const reading = getMockReading('XAUUSD', 'M15')!;
    const candles = getMockCandles('XAUUSD', 'M15')!;
    const lo = Math.min(...candles.map((c) => c.low));
    const hi = Math.max(...candles.map((c) => c.high));
    const s = reading.structure;
    const levels = [
      s.bos?.level,
      s.choch?.level,
      s.retest_in_progress?.level,
      ...s.order_blocks.flatMap((o) => [o.level_high, o.level_low]),
      ...s.fair_value_gaps.flatMap((f) => [f.level_high, f.level_low]),
    ].filter((x): x is number => typeof x === 'number');
    for (const level of levels) {
      expect(level).toBeGreaterThanOrEqual(lo);
      expect(level).toBeLessThanOrEqual(hi);
    }
  });

  it('has strictly increasing timestamps with valid OHLC ordering', () => {
    const candles = getMockCandles('EURUSD', 'M15')!;
    for (let i = 0; i < candles.length; i += 1) {
      const c = candles[i]!;
      expect(c.high).toBeGreaterThanOrEqual(c.low);
      expect(c.high).toBeGreaterThanOrEqual(Math.max(c.open, c.close));
      expect(c.low).toBeLessThanOrEqual(Math.min(c.open, c.close));
      if (i > 0) expect(c.time).toBeGreaterThan(candles[i - 1]!.time);
    }
  });

  it('marks XAU/USD H4 candle feed as unavailable (graceful-degradation demo)', () => {
    // Reading stays available, but the chart feed is intentionally absent.
    expect(getMockReading('XAUUSD', 'H4')).not.toBeNull();
    expect(getMockCandles('XAUUSD', 'H4')).toBeNull();
  });

  it('returns null candles for an out-of-catalogue combo', () => {
    expect(getMockCandles('BTCUSD', 'H1')).toBeNull();
  });
});
