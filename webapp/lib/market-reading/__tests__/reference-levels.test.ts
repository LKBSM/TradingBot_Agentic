import { describe, it, expect } from 'vitest';
import type { Candle, LiquidityPool } from '@/types/market-reading';
import {
  structureRange,
  positionPct,
  referenceLevels,
  distancePct,
} from '../reference-levels';

function candle(partial: Partial<Candle> & { time: number }): Candle {
  return { open: 0, high: 0, low: 0, close: 0, volume: 0, ...partial };
}

function pool(kind: LiquidityPool['kind'], level: number): LiquidityPool {
  return {
    id: `LIQ_${kind}_${level}`,
    side: kind === 'range_high' ? 'bsl' : 'ssl',
    kind,
    level,
    touches: 1,
    is_external: true,
    status: 'intact',
    created_at: '2026-07-26T00:00:00Z',
    user_flagged: false,
  };
}

describe('structureRange', () => {
  it('reads range_high / range_low from liquidity pools', () => {
    const r = structureRange({
      liquidity_pools: [pool('range_low', 2370.5), pool('range_high', 2421.2)],
    });
    expect(r).toEqual({ low: 2370.5, high: 2421.2 });
  });

  it('returns null when a bound is missing', () => {
    expect(structureRange({ liquidity_pools: [pool('range_high', 2421.2)] })).toBeNull();
    expect(structureRange({ liquidity_pools: [] })).toBeNull();
    expect(structureRange({})).toBeNull();
  });

  it('returns null on a degenerate range (high <= low)', () => {
    expect(
      structureRange({
        liquidity_pools: [pool('range_low', 2400), pool('range_high', 2400)],
      }),
    ).toBeNull();
  });

  it('falls back to external clusters when range_high/low are not emitted (RG-1b)', () => {
    const mk = (side: 'bsl' | 'ssl', kind: LiquidityPool['kind'], level: number, ext: boolean): LiquidityPool => ({
      id: `${side}_${level}`, side, kind, level, touches: 2, is_external: ext,
      status: 'intact', created_at: '2026-07-26T00:00:00Z', user_flagged: false,
    });
    const r = structureRange({
      liquidity_pools: [
        mk('bsl', 'equal_highs', 2421.2, true), // top external cluster
        mk('ssl', 'equal_lows', 2370.5, true), // bottom external cluster
        mk('bsl', 'equal_highs', 2410.0, false), // internal → ignored
      ],
    });
    expect(r).toEqual({ low: 2370.5, high: 2421.2 });
  });
});

describe('positionPct', () => {
  it('places the price between the bounds as 0–100 %', () => {
    expect(positionPct(2370.5, 2421.2, 2392.35)).toBeCloseTo(43.09, 1);
    expect(positionPct(2370.5, 2421.2, 2370.5)).toBe(0);
    expect(positionPct(2370.5, 2421.2, 2421.2)).toBe(100);
  });

  it('clamps a price outside the range to an edge', () => {
    expect(positionPct(2370.5, 2421.2, 2500)).toBe(100);
    expect(positionPct(2370.5, 2421.2, 2000)).toBe(0);
  });

  it('returns null on a degenerate range', () => {
    expect(positionPct(2400, 2400, 2400)).toBeNull();
    expect(positionPct(2400, 2390, 2395)).toBeNull();
  });
});

describe('referenceLevels', () => {
  const daily: Candle[] = [
    candle({ time: 1, open: 2410, high: 2424.8, low: 2370.5, close: 2415 }),
    candle({ time: 2, open: 2415, high: 2421.2, low: 2376.2, close: 2400 }), // prev day
    candle({ time: 3, open: 2398.6, high: 2405, low: 2390, close: 2392 }), // current day
  ];
  const weekly: Candle[] = [
    candle({ time: 10, open: 2380, high: 2450, low: 2350, close: 2420 }),
    candle({ time: 20, open: 2420, high: 2424.8, low: 2370.5, close: 2405 }), // prev week
    candle({ time: 30, open: 2411.4, high: 2412, low: 2390, close: 2392 }), // current week
  ];

  it('takes day/week open from the current (last) candle', () => {
    const r = referenceLevels(daily, weekly);
    expect(r.dayOpen).toBe(2398.6);
    expect(r.weekOpen).toBe(2411.4);
  });

  it('takes previous extremes from the second-to-last candle', () => {
    const r = referenceLevels(daily, weekly);
    expect(r.prevDayHigh).toBe(2421.2);
    expect(r.prevDayLow).toBe(2376.2);
    expect(r.prevWeekHigh).toBe(2424.8);
    expect(r.prevWeekLow).toBe(2370.5);
  });

  it('yields nulls for a series with no previous candle', () => {
    const r = referenceLevels([candle({ time: 1, open: 100 })], []);
    expect(r.dayOpen).toBe(100);
    expect(r.prevDayHigh).toBeNull();
    expect(r.prevDayLow).toBeNull();
    expect(r.weekOpen).toBeNull();
    expect(r.prevWeekHigh).toBeNull();
  });
});

describe('distancePct', () => {
  it('is the signed distance from price to the level', () => {
    expect(distancePct(2398.6, 2392.35)).toBeCloseTo(0.261, 2);
    expect(distancePct(2370.5, 2392.35)).toBeCloseTo(-0.913, 2);
  });

  it('returns null on a non-positive price', () => {
    expect(distancePct(2400, 0)).toBeNull();
  });
});
