import { describe, it, expect } from 'vitest';
import type { LiquidityPool } from '@/types/market-reading';
import {
  coincidenceTolerance,
  matchLiquidity,
  COINCIDENCE_FRACTION,
} from '../reference-coincidence';

function pool(p: Partial<LiquidityPool> & { level: number; side: LiquidityPool['side'] }): LiquidityPool {
  return {
    id: `p_${p.level}`,
    kind: p.side === 'bsl' ? 'equal_highs' : 'equal_lows',
    touches: 1,
    is_external: true,
    status: 'intact',
    created_at: '2026-07-27T00:00:00Z',
    user_flagged: false,
    ...p,
  };
}

describe('coincidenceTolerance', () => {
  it('is a quarter of the recent average candle amplitude', () => {
    expect(COINCIDENCE_FRACTION).toBe(0.25);
    expect(coincidenceTolerance(4)).toBe(1);
    expect(coincidenceTolerance(3.42)).toBeCloseTo(0.855, 3);
  });

  it('returns null without a usable amplitude basis (no coincidence claimed)', () => {
    expect(coincidenceTolerance(null)).toBeNull();
    expect(coincidenceTolerance(undefined)).toBeNull();
    expect(coincidenceTolerance(0)).toBeNull();
    expect(coincidenceTolerance(-2)).toBeNull();
  });
});

describe('matchLiquidity', () => {
  const pools: LiquidityPool[] = [
    pool({ level: 2421.2, side: 'bsl' }),
    pool({ level: 2370.5, side: 'ssl' }),
  ];
  const tol = coincidenceTolerance(3.42)!; // 0.855

  it('returns the side of a detected pocket within tolerance', () => {
    expect(matchLiquidity(2421.2, pools, tol)).toBe('bsl'); // exact
    expect(matchLiquidity(2421.9, pools, tol)).toBe('bsl'); // 0.7 ≤ 0.855
    expect(matchLiquidity(2370.5, pools, tol)).toBe('ssl');
  });

  it('returns null when no pocket is within tolerance (a coincidence, not an equivalence)', () => {
    expect(matchLiquidity(2398.6, pools, tol)).toBeNull(); // far from both
    expect(matchLiquidity(2422.2, pools, tol)).toBeNull(); // 1.0 > 0.855
  });

  it('never claims a coincidence without a tolerance basis', () => {
    expect(matchLiquidity(2421.2, pools, null)).toBeNull();
  });

  it('excludes broken pockets — they no longer rest at their level', () => {
    const broken = [pool({ level: 2421.2, side: 'bsl', status: 'broken' })];
    expect(matchLiquidity(2421.2, broken, tol)).toBeNull();
  });

  it('picks the nearest pocket when several are within tolerance', () => {
    const near = [
      pool({ level: 2421.2, side: 'bsl' }),
      pool({ level: 2421.0, side: 'ssl' }), // closer to 2421.05
    ];
    expect(matchLiquidity(2421.05, near, coincidenceTolerance(4)!)).toBe('ssl');
  });
});
