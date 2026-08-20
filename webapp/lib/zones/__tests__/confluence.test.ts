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

  // ── Id lock (mission §4): every ZONE fact carries the RELATED zone's REAL
  // engine id, threaded straight through — never reconstructed from price/label.
  // A liquidity pocket is not a zone, so it carries no id. ──
  it('carries the same-timeframe zone REAL id (raw, timeframe null)', () => {
    const inner = zl({ id: 'ob-inner-42', levelLow: 2373, levelHigh: 2376 });
    const facts = buildConfluence(subject, [subject, inner], [], []);
    expect(facts[0]).toMatchObject({ relation: 'inner', id: 'ob-inner-42', timeframe: null });
  });

  it('carries the sibling zone RAW engine id + its timeframe (no composite)', () => {
    const sib: SiblingZone = { id: 'fvg-h1-7', kind: 'ob', direction: 'bullish', levelLow: 2365, levelHigh: 2385, timeframe: 'H1' };
    const facts = buildConfluence(subject, [subject], [sib], []);
    // The id is the raw `fvg-h1-7`, NOT `H1-fvg-h1-7` — navigation matches the
    // engine id on the H1 reading, disambiguated by `timeframe`.
    expect(facts[0]).toMatchObject({ relation: 'outer', id: 'fvg-h1-7', timeframe: 'H1' });
  });

  it('does not confuse two siblings of equal price band — each keeps its own id', () => {
    const a: SiblingZone = { id: 'zone-A', kind: 'ob', direction: 'bullish', levelLow: 2372, levelHigh: 2378, timeframe: 'H1' };
    const b: SiblingZone = { id: 'zone-B', kind: 'fvg', direction: 'bearish', levelLow: 2372, levelHigh: 2378, timeframe: 'H4' };
    const facts = buildConfluence(subject, [subject], [a, b], []);
    const byTf = Object.fromEntries(facts.filter((f) => f.timeframe).map((f) => [f.timeframe, f.id]));
    expect(byTf).toEqual({ H1: 'zone-A', H4: 'zone-B' });
  });

  it('never attaches an id to a liquidity fact (a pool is not a navigable zone)', () => {
    const pool: LiquidityPool = {
      id: 'liq', side: 'bsl', kind: 'equal_highs', level: 2385, touches: 2,
      is_external: true, status: 'intact', created_at: 'x', user_flagged: false,
    };
    const facts = buildConfluence(subject, [subject], [], [pool]);
    const liq = facts.find((f) => f.relation === 'liquidity')!;
    expect(liq.id).toBeUndefined();
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
