import { describe, expect, it } from 'vitest';
import type { Candle, FairValueGap, OrderBlock, ZoneContact } from '@/types/market-reading';
import {
  barsSince,
  buildContactTimeline,
  buildTimeline,
  collectConsumedZones,
  collectZones,
  contactCount,
  fillFraction,
  formatDurationShort,
  fvgContactFills,
  isConsumed,
  matchesFilter,
  priceRelation,
  sortZones,
  zoneHeaderState,
  zonePositionGroup,
  zoneProximity,
  type ContactTimelineLabels,
  type DurationLabels,
  type TimelineLabels,
} from '../lifecycle';

const FR_TIMELINE: TimelineLabels = {
  formed: 'Formé',
  mitigated: 'Mitigé',
  obTested: 'Testé',
  fvgTested: 'Pénétré',
  filled: 'Comblé',
  partial: 'Partiellement comblé',
  active: 'Suivi en cours',
};

const FR_DURATION: DurationLabels = {
  underMinute: "moins d'une minute",
  min: 'min',
  hour: 'h',
  day: 'j',
};

const CT_LABELS: ContactTimelineLabels = {
  formed: 'Formée',
  entry: (n, total) => (total > 1 ? `Entrée ${n}` : 'Entrée'),
  touch: 'Touche',
  traversal: 'Traversée',
  now: 'Maintenant',
};

function contact(over: Partial<ZoneContact> = {}): ZoneContact {
  return { at: '2026-05-26T09:00:00+00:00', level: 2376, outcome: 'entry_exit', ...over };
}

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

function one(o: Partial<OrderBlock>) {
  return collectZones({ order_blocks: [ob(o)], fair_value_gaps: [] } as never)[0]!;
}

describe('collectZones', () => {
  it('projects OB + FVG with their real fields incl. contacts/origin', () => {
    const zones = collectZones({ order_blocks: [ob()], fair_value_gaps: [fvg()] } as never);
    expect(zones).toHaveLength(2);
    const o = zones.find((z) => z.kind === 'ob')!;
    expect(o).toMatchObject({ id: 'ob-1', isActive: true, isMitigated: false });
    expect(o.contacts).toEqual([]);
    expect(o.origin).toBeNull();
  });

  it('returns [] for a missing structure (graceful)', () => {
    expect(collectZones(null)).toEqual([]);
    expect(collectZones(undefined)).toEqual([]);
  });
});

describe('collectConsumedZones', () => {
  it('projects the separate consumed lists (the « Comblées » group)', () => {
    const zones = collectConsumedZones({
      order_blocks: [],
      fair_value_gaps: [],
      consumed_order_blocks: [ob({ id: 'c1', status: 'invalidated' })],
      consumed_fair_value_gaps: [fvg({ id: 'c2', status: 'filled' })],
    } as never);
    expect(zones.map((z) => z.id).sort()).toEqual(['c1', 'c2']);
    expect(zones.every(isConsumed)).toBe(true);
  });
});

// ── The three distinct contact states (mission question B) ──
describe('contact ledger — the three outcomes are distinct, never conflated', () => {
  const contacts: ZoneContact[] = [
    { at: '2026-05-26T09:00:00+00:00', level: 2376, outcome: 'entry_exit' },
    { at: '2026-05-26T09:30:00+00:00', level: 2377.9, outcome: 'edge_touch' },
  ];
  const z = one({ tested: true, contacts });

  it('contactCount counts completed contacts (edge + entry), excluding inside/traversal', () => {
    expect(contactCount(z)).toBe(2);
    const consumed = one({ status: 'invalidated', contacts: [...contacts, { at: 'x', level: 2375, outcome: 'traversal' }] });
    expect(contactCount(consumed)).toBe(2); // the traversal is not a "contact" tally
  });

  it('the timeline renders one node PER contact, labelled by outcome', () => {
    const t = buildContactTimeline(z, CT_LABELS);
    expect(t.map((e) => e.label)).toEqual(['Formée', 'Entrée', 'Touche', 'Maintenant']);
  });

  it('a consumed zone ends on the Traversée node (no « Maintenant »)', () => {
    const consumed = one({
      status: 'invalidated',
      contacts: [contact(), { at: '2026-05-26T10:00:00+00:00', level: 2375, outcome: 'traversal' }],
    });
    const t = buildContactTimeline(consumed, CT_LABELS);
    expect(t.map((e) => e.label)).toEqual(['Formée', 'Entrée', 'Traversée']);
  });

  it('numbers multiple entries', () => {
    const two = one({
      contacts: [
        { at: 'a', level: 2376, outcome: 'entry_exit' },
        { at: 'b', level: 2376, outcome: 'entry_exit' },
      ],
    });
    expect(buildContactTimeline(two, CT_LABELS).map((e) => e.label)).toEqual([
      'Formée',
      'Entrée 1',
      'Entrée 2',
      'Maintenant',
    ]);
  });
});

describe('zoneHeaderState — a fact, never a judgement', () => {
  it('untouched when nothing has contacted it', () => {
    expect(zoneHeaderState(one({}))).toEqual({ key: 'untouched' });
  });
  it('never_filled when the price is inside now but never consumed', () => {
    const z = one({ contacts: [{ at: 'x', level: 2376, outcome: 'inside' }] });
    expect(zoneHeaderState(z)).toEqual({ key: 'never_filled' });
  });
  it('contacts with a count when touched', () => {
    const z = one({ tested: true, contacts: [contact(), contact()] });
    expect(zoneHeaderState(z)).toEqual({ key: 'contacts', count: 2 });
  });
  it('consumed for a traversed/filled zone', () => {
    expect(zoneHeaderState(one({ status: 'invalidated' }))).toEqual({ key: 'consumed' });
  });
});

describe('zoneProximity — distance always carries a unit AND a reference edge', () => {
  const zone = one({ level_low: 2375, level_high: 2378 });

  it('inside: distance to each edge', () => {
    expect(zoneProximity(zone, 2376.5)).toEqual({ inside: true, distToLow: 1.5, distToHigh: 1.5 });
  });
  it('zone above the price: measured at the LOW edge, with % of price', () => {
    const p = zoneProximity(one({ level_low: 2385, level_high: 2388 }), 2380);
    expect(p).toMatchObject({ inside: false, side: 'above', edge: 'low' });
    expect((p as { distance: number }).distance).toBeCloseTo(5, 5);
    expect((p as { distancePct: number }).distancePct).toBeCloseTo(5 / 2380, 6);
  });
  it('zone below the price: measured at the HIGH edge', () => {
    const p = zoneProximity(zone, 2390);
    expect(p).toMatchObject({ inside: false, side: 'below', edge: 'high' });
    expect((p as { distance: number }).distance).toBeCloseTo(12, 5);
  });
  it('null without a usable price (never guessed)', () => {
    expect(zoneProximity(zone, null)).toBeNull();
    expect(zoneProximity(zone, Number.NaN)).toBeNull();
  });
});

describe('zonePositionGroup — dedans / au-dessus / en dessous', () => {
  it('classifies by the price', () => {
    expect(zonePositionGroup(one({ level_low: 2375, level_high: 2378 }), 2376)).toBe('inside');
    expect(zonePositionGroup(one({ level_low: 2385, level_high: 2388 }), 2380)).toBe('above');
    expect(zonePositionGroup(one({ level_low: 2360, level_high: 2365 }), 2380)).toBe('below');
  });
});

describe('fvgContactFills — cumulative comblement per contact', () => {
  it('runs the deepest penetration monotonically (bullish gap)', () => {
    const z = collectZones({
      order_blocks: [],
      fair_value_gaps: [
        fvg({
          level_low: 2378,
          level_high: 2381, // span 3
          status: 'partially_filled',
          contacts: [
            { at: 'a', level: 2380, outcome: 'entry_exit' }, // 1/3 filled
            { at: 'b', level: 2379, outcome: 'entry_exit' }, // 2/3 filled (deeper)
          ],
        }),
      ],
    } as never)[0]!;
    const fills = fvgContactFills(z);
    expect(fills[0]).toBeCloseTo(1 / 3, 5);
    expect(fills[1]).toBeCloseTo(2 / 3, 5);
  });
});

describe('matchesFilter — all / active / untouched / consumed', () => {
  const active = one({ status: 'active' });
  const touched = one({ status: 'active', tested: true, contacts: [contact()] });
  const consumed = one({ status: 'invalidated' });

  it('untouched keeps only never-contacted live zones', () => {
    expect(matchesFilter(active, 'untouched')).toBe(true);
    expect(matchesFilter(touched, 'untouched')).toBe(false);
    expect(matchesFilter(consumed, 'untouched')).toBe(false);
  });
  it('active excludes consumed', () => {
    expect(matchesFilter(active, 'active')).toBe(true);
    expect(matchesFilter(consumed, 'active')).toBe(false);
  });
  it('consumed keeps only consumed', () => {
    expect(matchesFilter(consumed, 'consumed')).toBe(true);
    expect(matchesFilter(active, 'consumed')).toBe(false);
  });
  it('all keeps everything', () => {
    expect([active, touched, consumed].every((z) => matchesFilter(z, 'all'))).toBe(true);
  });
});

describe('sortZones — factual orders only (NO importance/quality sort)', () => {
  it('formation orders by created_at desc', () => {
    const zones = collectZones({
      order_blocks: [
        ob({ id: 'old', created_at: '2026-05-26T05:00:00+00:00' }),
        ob({ id: 'new', created_at: '2026-05-26T11:00:00+00:00' }),
      ],
      fair_value_gaps: [],
    } as never);
    expect(sortZones(zones, 'formation').map((z) => z.id)).toEqual(['new', 'old']);
  });

  it('contacts orders by contact count desc', () => {
    const zones = [
      one({ id: 'two', contacts: [contact(), contact()] }),
      one({ id: 'none' }),
      one({ id: 'one', contacts: [contact()] }),
    ];
    expect(sortZones(zones, 'contacts').map((z) => z.id)).toEqual(['two', 'one', 'none']);
  });

  it('proximity ignores importance — a low-importance NEAR zone beats a high-importance FAR one', () => {
    const zones = collectZones({
      order_blocks: [
        ob({ id: 'far-high', importance: 'high', level_low: 2400, level_high: 2402 }),
        ob({ id: 'near-low', importance: 'low', level_low: 2393, level_high: 2394 }),
      ],
      fair_value_gaps: [],
    } as never);
    expect(sortZones(zones, 'proximity', 2391).map((z) => z.id)).toEqual(['near-low', 'far-high']);
  });
});

// ── Retained coverage of unchanged helpers ──
describe('buildTimeline (status-based) still honest', () => {
  it('active untested OB: Formé → Suivi en cours', () => {
    const t = buildTimeline(one({ tested: false }), FR_TIMELINE);
    expect(t.map((e) => e.key)).toEqual(['formed', 'active']);
  });
});

describe('fillFraction', () => {
  it('bullish gap: (high − fill) / span', () => {
    const z = collectZones({
      order_blocks: [],
      fair_value_gaps: [fvg({ level_low: 2378, level_high: 2381, fill_level: 2379.5 })],
    } as never)[0]!;
    expect(fillFraction(z)).toBeCloseTo(0.5, 5);
  });
});

describe('priceRelation', () => {
  const zone = one({ level_low: 2375, level_high: 2378 });
  it('inside / above / below', () => {
    expect(priceRelation(zone, 2376.5)).toEqual({ position: 'inside' });
    expect(priceRelation(zone, 2370)).toMatchObject({ position: 'above' });
    expect(priceRelation(zone, 2392)).toMatchObject({ position: 'below' });
  });
});

describe('barsSince', () => {
  const t0 = Date.parse('2026-05-26T10:00:00+00:00') / 1000;
  const candles: Candle[] = Array.from({ length: 5 }, (_, i) => ({
    time: t0 + i * 900,
    open: 1,
    high: 2,
    low: 0,
    close: 1,
  }));
  it('counts candles after formation, null when truncated', () => {
    expect(barsSince(candles, '2026-05-26T10:00:00+00:00')).toBe(4);
    expect(barsSince(candles, '2026-05-26T08:00:00+00:00')).toBeNull();
  });
});

describe('formatDurationShort', () => {
  it('compact minutes/hours/days', () => {
    expect(formatDurationShort(45 * 60 * 1000, FR_DURATION)).toBe('45 min');
    expect(formatDurationShort((6 * 60 + 30) * 60 * 1000, FR_DURATION)).toBe('6 h 30');
  });
});
