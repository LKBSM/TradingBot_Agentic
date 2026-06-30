import { describe, expect, it } from 'vitest';
import type { FairValueGap, OrderBlock } from '@/types/market-reading';
import {
  buildTimeline,
  collectZones,
  fillFraction,
  matchesFilter,
  narrateZone,
  sortZones,
  type ZoneLifecycle,
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

describe('sortZones', () => {
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

  it('proximity orders by distance of the band mid to price', () => {
    const zones = collectZones({
      order_blocks: [
        ob({ id: 'far', level_low: 2400, level_high: 2402 }),
        ob({ id: 'near', level_low: 2390, level_high: 2392 }),
      ],
      fair_value_gaps: [],
    } as never);
    expect(sortZones(zones, 'proximity', 2391).map((z) => z.id)).toEqual(['near', 'far']);
  });
});

describe('narrateZone — factual, never predictive', () => {
  const cases: ZoneLifecycle[] = [
    collectZones({ order_blocks: [ob({ status: 'active', tested: false })], fair_value_gaps: [] } as never)[0]!,
    collectZones({ order_blocks: [ob({ status: 'active', tested: true, mitigated_at: '2026-05-26T09:00:00+00:00' })], fair_value_gaps: [] } as never)[0]!,
    collectZones({ order_blocks: [ob({ status: 'mitigated', mitigated_at: '2026-05-26T08:30:00+00:00' })], fair_value_gaps: [] } as never)[0]!,
    collectZones({ order_blocks: [], fair_value_gaps: [fvg({ status: 'active' })] } as never)[0]!,
    collectZones({ order_blocks: [], fair_value_gaps: [fvg({ status: 'partially_filled', fill_level: 2379.5 })] } as never)[0]!,
    collectZones({ order_blocks: [], fair_value_gaps: [fvg({ status: 'filled' })] } as never)[0]!,
  ];

  it('produces present/past descriptive sentences', () => {
    expect(narrateZone(cases[0]!, 'XAUUSD')).toContain('Order Block');
    expect(narrateZone(cases[0]!, 'XAUUSD')).toContain('Non testé');
    expect(narrateZone(cases[2]!, 'XAUUSD')).toContain('Mitigé');
    expect(narrateZone(cases[3]!, 'XAUUSD')).toContain('Intact');
    expect(narrateZone(cases[4]!, 'XAUUSD')).toMatch(/Partiellement comblé/);
    expect(narrateZone(cases[5]!, 'XAUUSD')).toContain('comblé');
  });

  it('contains no predictive or directive vocabulary', () => {
    // Forbidden lexicon — chosen NOT to overlap with legitimate descriptive
    // words (haussier/baissier/mitigé/comblé/testé/pénétré/actif/importance).
    const FORBIDDEN =
      /(\bva\b|vise|cible|objectif|devrait|rebond|prévis|prédi|acheter|vendre|\bsignal\b|probab|attendu|recommand)/i;
    for (const z of cases) {
      expect(narrateZone(z, 'XAUUSD')).not.toMatch(FORBIDDEN);
    }
  });
});
