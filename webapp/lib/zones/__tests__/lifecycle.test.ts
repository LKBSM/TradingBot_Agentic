import { describe, expect, it } from 'vitest';
import type { Candle, FairValueGap, OrderBlock } from '@/types/market-reading';
import {
  barsSince,
  buildTimeline,
  collectZones,
  fillFraction,
  findOverlaps,
  formatDurationShort,
  matchesFilter,
  priceRelation,
  sortZones,
  type SiblingZone,
} from '../lifecycle';

function ob(overrides: Partial<OrderBlock> = {}): OrderBlock {
  return {
    id: 'ob-1',
    direction: 'bullish',
    level_high: 2378,
    level_low: 2375,
    importance: 'high',
    status: 'active',
    created_at: '2026-05-26T08:00:00+00:00',
    tested: false,
    user_flagged: false,
    ...overrides,
  };
}

function fvg(overrides: Partial<FairValueGap> = {}): FairValueGap {
  return {
    id: 'fvg-1',
    direction: 'bullish',
    level_high: 2381,
    level_low: 2378,
    status: 'active',
    created_at: '2026-05-26T10:45:00+00:00',
    tested: false,
    user_flagged: false,
    ...overrides,
  };
}

describe('collectZones', () => {
  it('projects OB + FVG with their real fields, no invented ones', () => {
    const zones = collectZones({
      order_blocks: [ob()],
      fair_value_gaps: [fvg()],
    } as never);
    expect(zones).toHaveLength(2);
    const o = zones.find((z) => z.kind === 'ob')!;
    expect(o).toMatchObject({
      id: 'ob-1',
      direction: 'bullish',
      importance: 'high',
      status: 'active',
      tested: false,
      isActive: true,
      isMitigated: false,
    });
    const f = zones.find((z) => z.kind === 'fvg')!;
    // FVG carries no importance — the engine emits none.
    expect(f.importance).toBeNull();
  });

  it('returns [] for a missing structure (graceful)', () => {
    expect(collectZones(null)).toEqual([]);
    expect(collectZones(undefined)).toEqual([]);
  });
});

describe('buildTimeline — only real events, never fabricated', () => {
  it('active + untested OB: Formé → Suivi en cours (no Testé step)', () => {
    const [z] = collectZones({ order_blocks: [ob({ tested: false })], fair_value_gaps: [] } as never);
    const t = buildTimeline(z!);
    expect(t.map((e) => e.key)).toEqual(['formed', 'active']);
    expect(t.some((e) => e.label === 'Testé')).toBe(false);
  });

  it('active + tested OB: adds a single Testé step (no count)', () => {
    const [z] = collectZones({
      order_blocks: [ob({ tested: true, mitigated_at: '2026-05-26T09:00:00+00:00' })],
      fair_value_gaps: [],
    } as never);
    const t = buildTimeline(z!);
    expect(t.map((e) => e.key)).toEqual(['formed', 'tested', 'active']);
    const tested = t.filter((e) => e.key === 'tested');
    expect(tested).toHaveLength(1); // never "×N"
    expect(tested[0]!.at).toBe('2026-05-26T09:00:00+00:00');
  });

  it('mitigated OB: terminal Mitigé step at mitigated_at', () => {
    const [z] = collectZones({
      order_blocks: [ob({ status: 'mitigated', tested: true, mitigated_at: '2026-05-26T08:30:00+00:00' })],
      fair_value_gaps: [],
    } as never);
    const t = buildTimeline(z!);
    expect(t.map((e) => e.key)).toEqual(['formed', 'mitigated']);
    expect(t[1]).toMatchObject({ variant: 'terminal', at: '2026-05-26T08:30:00+00:00' });
  });

  it('filled FVG with no fill timestamp degrades to a dateless terminal step', () => {
    const [z] = collectZones({
      order_blocks: [],
      fair_value_gaps: [fvg({ status: 'filled', tested: false })],
    } as never);
    const t = buildTimeline(z!);
    expect(t.map((e) => e.key)).toEqual(['formed', 'filled']);
    // No "filled_at" exists → we surface the step WITHOUT inventing a date.
    expect(t[1]!.at).toBeNull();
  });

  it('partially_filled FVG: partial step then ongoing', () => {
    const [z] = collectZones({
      order_blocks: [],
      fair_value_gaps: [fvg({ status: 'partially_filled', tested: true, mitigated_at: '2026-05-26T09:15:00+00:00' })],
    } as never);
    const t = buildTimeline(z!);
    expect(t.map((e) => e.key)).toEqual(['formed', 'partial', 'active']);
  });
});

describe('fillFraction — derived from the real fill_level price', () => {
  it('bullish gap fills from the top: (high − fill) / span', () => {
    const [z] = collectZones({
      order_blocks: [],
      fair_value_gaps: [fvg({ level_low: 2378, level_high: 2381, fill_level: 2379.5 })],
    } as never);
    expect(fillFraction(z!)).toBeCloseTo(0.5, 5);
  });

  it('bearish gap fills from the bottom: (fill − low) / span', () => {
    const [z] = collectZones({
      order_blocks: [],
      fair_value_gaps: [fvg({ direction: 'bearish', level_low: 2378, level_high: 2381, fill_level: 2379.5 })],
    } as never);
    expect(fillFraction(z!)).toBeCloseTo(0.5, 5);
  });

  it('returns null without a fill_level (no invention)', () => {
    const [z] = collectZones({
      order_blocks: [],
      fair_value_gaps: [fvg({ status: 'partially_filled' })], // no fill_level
    } as never);
    expect(fillFraction(z!)).toBeNull();
  });
});

describe('matchesFilter', () => {
  it('maps Actives / Mitigées to OB and FVG statuses', () => {
    const activeOb = collectZones({ order_blocks: [ob({ status: 'active' })], fair_value_gaps: [] } as never)[0]!;
    const mitigatedOb = collectZones({ order_blocks: [ob({ status: 'mitigated' })], fair_value_gaps: [] } as never)[0]!;
    const filledFvg = collectZones({ order_blocks: [], fair_value_gaps: [fvg({ status: 'filled' })] } as never)[0]!;

    expect(matchesFilter(activeOb, 'active')).toBe(true);
    expect(matchesFilter(activeOb, 'mitigated')).toBe(false);
    expect(matchesFilter(mitigatedOb, 'mitigated')).toBe(true);
    expect(matchesFilter(filledFvg, 'mitigated')).toBe(true);
    expect(matchesFilter(activeOb, 'all')).toBe(true);
  });
});

describe('priceRelation — present-tense geometric fact', () => {
  const zone = collectZones({
    order_blocks: [ob({ level_low: 2375, level_high: 2378 })],
    fair_value_gaps: [],
  } as never)[0]!;

  it('inside when the price is within the band (edges included)', () => {
    expect(priceRelation(zone, 2376.5)).toEqual({ position: 'inside' });
    expect(priceRelation(zone, 2375)).toEqual({ position: 'inside' });
    expect(priceRelation(zone, 2378)).toEqual({ position: 'inside' });
  });

  it('below the price: gap measured to the NEAREST edge (high)', () => {
    expect(priceRelation(zone, 2392.35)).toEqual({
      position: 'below',
      distance: expect.closeTo(14.35, 5) as never,
    });
  });

  it('above the price: gap measured to the nearest edge (low)', () => {
    expect(priceRelation(zone, 2370)).toEqual({
      position: 'above',
      distance: expect.closeTo(5, 5) as never,
    });
  });

  it('returns null without a usable price (badge omitted, never guessed)', () => {
    expect(priceRelation(zone, null)).toBeNull();
    expect(priceRelation(zone, undefined)).toBeNull();
    expect(priceRelation(zone, Number.NaN)).toBeNull();
  });
});

describe('barsSince — counted on real candles, never estimated', () => {
  // Ascending M15 window: 10:00 → 11:00 (5 bars).
  const t0 = Date.parse('2026-05-26T10:00:00+00:00') / 1000;
  const candles: Candle[] = Array.from({ length: 5 }, (_, i) => ({
    time: t0 + i * 900,
    open: 1,
    high: 2,
    low: 0,
    close: 1,
  }));

  it('counts the candles strictly after the formation bar', () => {
    expect(barsSince(candles, '2026-05-26T10:00:00+00:00')).toBe(4);
    expect(barsSince(candles, '2026-05-26T10:30:00+00:00')).toBe(2);
  });

  it('0 when the formation bar is the last one', () => {
    expect(barsSince(candles, '2026-05-26T11:00:00+00:00')).toBe(0);
  });

  it('null when the window does not reach the formation (truncated count)', () => {
    expect(barsSince(candles, '2026-05-26T08:00:00+00:00')).toBeNull();
  });

  it('null without candles or on an unparsable timestamp', () => {
    expect(barsSince(null, '2026-05-26T10:00:00+00:00')).toBeNull();
    expect(barsSince([], '2026-05-26T10:00:00+00:00')).toBeNull();
    expect(barsSince(candles, 'not-a-date')).toBeNull();
  });
});

describe('formatDurationShort', () => {
  it('formats minutes, hours and days compactly', () => {
    expect(formatDurationShort(30 * 1000)).toBe("moins d'une minute");
    expect(formatDurationShort(45 * 60 * 1000)).toBe('45 min');
    expect(formatDurationShort((6 * 60 + 30) * 60 * 1000)).toBe('6 h 30');
    expect(formatDurationShort(6 * 60 * 60 * 1000)).toBe('6 h');
    expect(formatDurationShort((2 * 24 + 4) * 60 * 60 * 1000)).toBe('2 j 4 h');
    expect(formatDurationShort(2 * 24 * 60 * 60 * 1000)).toBe('2 j');
  });
});

describe('findOverlaps — pure interval intersection', () => {
  const zone = collectZones({
    order_blocks: [ob({ level_low: 2375, level_high: 2378 })],
    fair_value_gaps: [],
  } as never)[0]!;

  const sibling = (over: Partial<SiblingZone>): SiblingZone => ({
    id: 's-1',
    kind: 'ob',
    direction: 'bullish',
    levelHigh: 2380,
    levelLow: 2376,
    timeframe: 'H1',
    ...over,
  });

  it('keeps a sibling whose band intersects', () => {
    expect(findOverlaps(zone, [sibling({})])).toHaveLength(1);
  });

  it('drops a disjoint sibling and a merely-touching edge', () => {
    expect(findOverlaps(zone, [sibling({ levelLow: 2380, levelHigh: 2384 })])).toHaveLength(0);
    // Touching at exactly 2378 is not an overlap (zero-width intersection).
    expect(findOverlaps(zone, [sibling({ levelLow: 2378, levelHigh: 2384 })])).toHaveLength(0);
  });
});

describe('sortZones — factual orders only (no importance/quality sort)', () => {
  it('recency orders by created_at desc', () => {
    const zones = collectZones({
      order_blocks: [
        ob({ id: 'old', created_at: '2026-05-26T05:00:00+00:00' }),
        ob({ id: 'new', created_at: '2026-05-26T11:00:00+00:00' }),
      ],
      fair_value_gaps: [],
    } as never);
    expect(sortZones(zones, 'recency').map((z) => z.id)).toEqual(['new', 'old']);
  });

  it('proximity orders by distance to the BAND, price-inside first', () => {
    const zones = collectZones({
      order_blocks: [
        ob({ id: 'far', level_low: 2400, level_high: 2402 }),
        ob({ id: 'inside', level_low: 2390, level_high: 2392 }),
        ob({ id: 'near', level_low: 2393, level_high: 2394 }),
      ],
      fair_value_gaps: [],
    } as never);
    expect(sortZones(zones, 'proximity', 2391).map((z) => z.id)).toEqual([
      'inside',
      'near',
      'far',
    ]);
  });

  it('proximity without a price degrades to the state order (no invented distance)', () => {
    const zones = collectZones({
      order_blocks: [ob({ id: 'mit', status: 'mitigated' }), ob({ id: 'act', status: 'active' })],
      fair_value_gaps: [],
    } as never);
    expect(sortZones(zones, 'proximity', null).map((z) => z.id)).toEqual(['act', 'mit']);
  });

  it('state follows the engine lifecycle status', () => {
    const zones = collectZones({
      order_blocks: [ob({ id: 'mit', status: 'mitigated' }), ob({ id: 'act', status: 'active' })],
      fair_value_gaps: [{ ...fvg({ id: 'part', status: 'partially_filled' }) }],
    } as never);
    expect(sortZones(zones, 'state').map((z) => z.id)).toEqual(['act', 'part', 'mit']);
  });
});
