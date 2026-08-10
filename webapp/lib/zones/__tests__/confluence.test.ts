import { describe, expect, it } from 'vitest';
import type { LiquidityPool } from '@/types/market-reading';
import { buildConfluence } from '../confluence';
import type { SiblingZone, ZoneLifecycle } from '../lifecycle';

function zl(over: Partial<ZoneLifecycle> = {}): ZoneLifecycle {
  return {
    id: 'z',
    kind: 'ob',
    direction: 'bullish',
    levelHigh: 2380,
    levelLow: 2370,
    importance: 'high',
    status: 'active',
    createdAt: '2026-05-26T08:00:00+00:00',
    tested: false,
    mitigatedAt: null,
    fillLevel: null,
    isActive: true,
    isMitigated: false,
    contacts: [],
    origin: null,
    session: null,
    ...over,
  };
}

const subject = zl({ id: 'subject', levelLow: 2370, levelHigh: 2380 });

describe('buildConfluence — containment vocabulary, never « chevauche »', () => {
  it('labels a same-timeframe smaller zone as inner', () => {
    const inner = zl({ id: 'inner', levelLow: 2373, levelHigh: 2376 });
    const facts = buildConfluence(subject, [subject, inner], [], []);
    expect(facts).toHaveLength(1);
    expect(facts[0]).toMatchObject({ relation: 'inner', timeframe: null });
  });

  it('labels a bigger other-timeframe zone as outer, keeping its unit name', () => {
    const sib: SiblingZone = { id: 's', kind: 'ob', direction: 'bullish', levelLow: 2365, levelHigh: 2385, timeframe: 'H1' };
    const facts = buildConfluence(subject, [subject], [sib], []);
    expect(facts[0]).toMatchObject({ relation: 'outer', timeframe: 'H1' });
  });

  it('labels a partial overlap as same_level (not containment)', () => {
    const sib: SiblingZone = { id: 's', kind: 'fvg', direction: 'bearish', levelLow: 2378, levelHigh: 2390, timeframe: 'H4' };
    const facts = buildConfluence(subject, [subject], [sib], []);
    expect(facts.find((f) => f.timeframe === 'H4')).toMatchObject({ relation: 'same_level' });
  });

  it('surfaces a nearby liquidity pocket with its distance + side', () => {
    const pool: LiquidityPool = {
      id: 'liq', side: 'bsl', kind: 'equal_highs', level: 2385, touches: 2,
      is_external: true, status: 'intact', created_at: 'x', user_flagged: false,
    };
    const facts = buildConfluence(subject, [subject], [], [pool]);
    const liq = facts.find((f) => f.relation === 'liquidity')!;
    expect(liq).toMatchObject({ liquiditySide: 'bsl', distanceSide: 'above' });
    expect(liq.distance).toBeCloseTo(5, 5);
  });

  it('returns [] when nothing is at this level (→ the card shows the absence state)', () => {
    const farSib: SiblingZone = { id: 'f', kind: 'ob', direction: 'bullish', levelLow: 2500, levelHigh: 2510, timeframe: 'H1' };
    const farPool: LiquidityPool = {
      id: 'p', side: 'ssl', kind: 'equal_lows', level: 2000, touches: 1,
      is_external: true, status: 'intact', created_at: 'x', user_flagged: false,
    };
    expect(buildConfluence(subject, [subject], [farSib], [farPool])).toEqual([]);
  });
});
