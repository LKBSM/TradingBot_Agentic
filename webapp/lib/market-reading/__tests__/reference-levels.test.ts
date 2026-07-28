import { describe, it, expect } from 'vitest';
import type { LiquidityPool, ReferenceLevelsPayload } from '@/types/market-reading';
import {
  structureRange,
  positionPct,
  referenceLevelsFromPayload,
  distancePct,
} from '../reference-levels';

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

describe('referenceLevelsFromPayload', () => {
  const payload: ReferenceLevelsPayload = {
    day_open: 2398.6,
    week_open: 2411.4,
    prev_day_high: 2421.2,
    prev_day_low: 2376.2,
    prev_week_high: 2424.8,
    prev_week_low: 2370.5,
    day_complete: true,
    week_complete: true,
  };

  it('maps the server-aggregated levels straight through (RG-1c)', () => {
    const r = referenceLevelsFromPayload(payload);
    expect(r).toEqual({
      dayOpen: 2398.6,
      weekOpen: 2411.4,
      prevDayHigh: 2421.2,
      prevDayLow: 2376.2,
      prevWeekHigh: 2424.8,
      prevWeekLow: 2370.5,
      dayComplete: true,
      weekComplete: true,
    });
  });

  it('carries per-period completeness so the panel can name « données insuffisantes »', () => {
    const r = referenceLevelsFromPayload({
      ...payload,
      prev_week_high: null,
      prev_week_low: null,
      week_complete: false,
    });
    expect(r?.dayComplete).toBe(true);
    expect(r?.weekComplete).toBe(false);
    expect(r?.prevWeekHigh).toBeNull();
    // A dropped period never shows a partial value — it arrives as null.
    expect(r?.prevWeekLow).toBeNull();
  });

  it('returns null when the payload is absent (static fixture / older backend)', () => {
    expect(referenceLevelsFromPayload(null)).toBeNull();
    expect(referenceLevelsFromPayload(undefined)).toBeNull();
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
