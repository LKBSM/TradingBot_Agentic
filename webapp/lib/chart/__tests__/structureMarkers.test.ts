import { describe, expect, it } from 'vitest';
import { buildStructureMarkers } from '../structureMarkers';
import type {
  BOSRecent,
  CHOCHRecent,
  MarketReadingStructure,
} from '@/types/market-reading';

const bos = (direction: 'bullish' | 'bearish', broken_at: string): BOSRecent => ({
  direction,
  level: 100,
  broken_at,
  validation_status: 'confirmed',
});
const choch = (
  direction: 'bullish' | 'bearish',
  broken_at: string,
): CHOCHRecent => ({ direction, level: 100, broken_at, validation_status: 'confirmed' });

function structure(
  bos_events: BOSRecent[] = [],
  choch_events: CHOCHRecent[] = [],
): MarketReadingStructure {
  return { bos: null, choch: null, bos_events, choch_events, order_blocks: [], fair_value_gaps: [] };
}

const T = (iso: string) => Math.floor(Date.parse(iso) / 1000);

describe('buildStructureMarkers', () => {
  it('returns nothing for an empty structure', () => {
    expect(buildStructureMarkers(structure())).toEqual([]);
  });

  it('marks each break, bullish below (↑) and bearish above (↓), sorted by time', () => {
    const m = buildStructureMarkers(
      structure([
        bos('bearish', '2026-05-28T05:00:00Z'),
        bos('bullish', '2026-05-28T02:00:00Z'),
      ]),
    );
    expect(m.map((x) => x.time)).toEqual([
      T('2026-05-28T02:00:00Z'),
      T('2026-05-28T05:00:00Z'),
    ]);
    expect(m[0]).toMatchObject({ shape: 'arrowUp', position: 'belowBar', text: 'BOS' });
    expect(m[1]).toMatchObject({ shape: 'arrowDown', position: 'aboveBar', text: 'BOS' });
  });

  it('CHOCH wins a shared bar (drops the duplicate BOS at that time)', () => {
    const ts = '2026-05-28T03:00:00Z';
    const m = buildStructureMarkers(
      structure([bos('bullish', ts)], [choch('bullish', ts)]),
    );
    expect(m).toHaveLength(1);
    expect(m[0]).toMatchObject({ text: 'CHOCH', time: T(ts) });
  });

  it('keeps BOS and CHOCH that fall on different bars', () => {
    const m = buildStructureMarkers(
      structure(
        [bos('bullish', '2026-05-28T01:00:00Z')],
        [choch('bearish', '2026-05-28T04:00:00Z')],
      ),
    );
    expect(m.map((x) => x.text)).toEqual(['BOS', 'CHOCH']);
  });

  it('skips events with an unparseable timestamp', () => {
    const m = buildStructureMarkers(structure([bos('bullish', 'not-a-date')]));
    expect(m).toEqual([]);
  });
});
