import { describe, expect, it } from 'vitest';
import { buildLiquidityLines } from '../liquidityLines';
import type {
  LiquidityPool,
  MarketReadingStructure,
} from '@/types/market-reading';

const pool = (over: Partial<LiquidityPool> = {}): LiquidityPool => ({
  id: 'liq_1',
  side: 'bsl',
  kind: 'equal_highs',
  level: 2000,
  touches: 2,
  is_external: true,
  status: 'intact',
  created_at: '2026-06-28T00:00:00Z',
  user_flagged: false,
  ...over,
});

function structure(
  liquidity_pools: LiquidityPool[] = [],
): MarketReadingStructure {
  return {
    bos: null,
    choch: null,
    bos_events: [],
    choch_events: [],
    order_blocks: [],
    fair_value_gaps: [],
    liquidity_pools,
  };
}

describe('buildLiquidityLines', () => {
  it('returns nothing when there are no pools (or the field is absent)', () => {
    expect(buildLiquidityLines(structure())).toEqual([]);
    expect(
      buildLiquidityLines({
        bos: null,
        choch: null,
        order_blocks: [],
        fair_value_gaps: [],
      } as MarketReadingStructure),
    ).toEqual([]);
  });

  it('maps a pool to a price line with side, status and a compact title', () => {
    const [line] = buildLiquidityLines(
      structure([pool({ id: 'liq_x', side: 'ssl', kind: 'range_low', level: 1980, status: 'swept' })]),
    );
    expect(line).toMatchObject({
      id: 'liq_x',
      price: 1980,
      side: 'ssl',
      status: 'swept',
      title: 'SSL · RL',
    });
    expect(line!.description).toContain('balayée');
  });

  it('excludes broken pools by default, includes them on request', () => {
    const s = structure([
      pool({ id: 'a', status: 'intact' }),
      pool({ id: 'b', status: 'broken' }),
    ]);
    expect(buildLiquidityLines(s).map((l) => l.id)).toEqual(['a']);
    expect(
      buildLiquidityLines(s, { includeBroken: true })
        .map((l) => l.id)
        .sort(),
    ).toEqual(['a', 'b']);
  });

  it('skips pools with a non-finite level', () => {
    expect(
      buildLiquidityLines(structure([pool({ level: Number.NaN })])),
    ).toEqual([]);
  });

  it('orders by status so intact lines are created last (drawn on top)', () => {
    const lines = buildLiquidityLines(
      structure([
        pool({ id: 'swept', status: 'swept' }),
        pool({ id: 'broken', status: 'broken' }),
        pool({ id: 'intact', status: 'intact' }),
      ]),
      { includeBroken: true },
    );
    expect(lines.map((l) => l.id)).toEqual(['broken', 'swept', 'intact']);
  });
});
