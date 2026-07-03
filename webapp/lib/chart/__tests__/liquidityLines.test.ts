import { describe, expect, it } from 'vitest';
import { buildLiquidityLines, poolContactSec } from '../liquidityLines';
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

const sec = (iso: string) => Math.floor(Date.parse(iso) / 1000);

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

  it('maps a pool to a bounded segment with side, status, anchors and labels', () => {
    const [line] = buildLiquidityLines(
      structure([
        pool({
          id: 'liq_x',
          side: 'ssl',
          kind: 'range_low',
          level: 1980,
          status: 'swept',
          created_at: '2026-06-27T00:00:00Z',
          swept_at: '2026-06-28T12:00:00Z',
        }),
      ]),
    );
    expect(line).toMatchObject({
      id: 'liq_x',
      price: 1980,
      side: 'ssl',
      status: 'swept',
      title: 'SSL · RL',
      createdSec: sec('2026-06-27T00:00:00Z'),
      contactSec: sec('2026-06-28T12:00:00Z'),
    });
    expect(line!.description).toContain('prise');
    expect(line!.chartLabel).toBe('Liquidité vente · prise');
  });

  it('an INTACT pocket has no contact point (extends to the current bar)', () => {
    const [line] = buildLiquidityLines(structure([pool({ status: 'intact' })]));
    expect(line!.contactSec).toBeNull();
    expect(line!.chartLabel).toBe('Liquidité achat · intacte');
  });

  it('includes BROKEN pockets by default (frozen segment, never removed)', () => {
    const s = structure([
      pool({ id: 'a', status: 'intact' }),
      pool({ id: 'b', status: 'broken', broken_at: '2026-06-29T00:00:00Z' }),
    ]);
    const lines = buildLiquidityLines(s);
    expect(lines.map((l) => l.id).sort()).toEqual(['a', 'b']);
    const broken = lines.find((l) => l.id === 'b')!;
    expect(broken.contactSec).toBe(sec('2026-06-29T00:00:00Z'));
    expect(broken.description).toContain('cassée');
  });

  it('intactOnly hides swept + broken segments (display filter, reversible)', () => {
    const s = structure([
      pool({ id: 'a', status: 'intact' }),
      pool({ id: 'b', status: 'swept', swept_at: '2026-06-29T00:00:00Z' }),
      pool({ id: 'c', status: 'broken', broken_at: '2026-06-29T04:00:00Z' }),
    ]);
    expect(buildLiquidityLines(s, { intactOnly: true }).map((l) => l.id)).toEqual(['a']);
    // Reversible: without the filter everything is back — nothing was lost.
    expect(buildLiquidityLines(s).map((l) => l.id).sort()).toEqual(['a', 'b', 'c']);
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
    );
    expect(lines.map((l) => l.id)).toEqual(['broken', 'swept', 'intact']);
  });
});

describe('poolContactSec', () => {
  it('is null while intact — even if stale timestamps linger', () => {
    expect(poolContactSec(pool({ status: 'intact' }))).toBeNull();
  });

  it('uses swept_at for a swept pocket and broken_at for a broken one', () => {
    expect(
      poolContactSec(pool({ status: 'swept', swept_at: '2026-06-28T12:00:00Z' })),
    ).toBe(sec('2026-06-28T12:00:00Z'));
    expect(
      poolContactSec(pool({ status: 'broken', broken_at: '2026-06-29T00:00:00Z' })),
    ).toBe(sec('2026-06-29T00:00:00Z'));
  });

  it('freezes at the FIRST contact when a broken pocket was swept earlier', () => {
    expect(
      poolContactSec(
        pool({
          status: 'broken',
          swept_at: '2026-06-28T12:00:00Z',
          broken_at: '2026-06-29T00:00:00Z',
        }),
      ),
    ).toBe(sec('2026-06-28T12:00:00Z'));
  });

  it('is null (defensive) when a swept/broken pocket has no parseable timestamp', () => {
    expect(poolContactSec(pool({ status: 'swept' }))).toBeNull();
    expect(poolContactSec(pool({ status: 'broken', broken_at: null }))).toBeNull();
  });
});
