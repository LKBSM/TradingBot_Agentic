import { describe, expect, it } from 'vitest';
import {
  timeframeToMinutes,
  nextCandleCloseMs,
  freshnessLabel,
} from '../candle-clock';

const at = (iso: string) => Date.parse(iso);

describe('timeframeToMinutes', () => {
  it('maps known timeframes (case-insensitive)', () => {
    expect(timeframeToMinutes('M15')).toBe(15);
    expect(timeframeToMinutes('h1')).toBe(60);
    expect(timeframeToMinutes('H4')).toBe(240);
  });
  it('returns null for an unknown timeframe', () => {
    expect(timeframeToMinutes('X9')).toBeNull();
  });
});

describe('nextCandleCloseMs', () => {
  it('aligns M15 to :00/:15/:30/:45 UTC', () => {
    const now = at('2026-06-30T12:07:00Z');
    expect(nextCandleCloseMs(['M15'], now)).toBe(at('2026-06-30T12:15:00Z'));
  });

  it('aligns H1 to the top of the hour and H4 to 0/4/8/12/16/20 UTC', () => {
    const now = at('2026-06-30T12:07:00Z');
    expect(nextCandleCloseMs(['H1'], now)).toBe(at('2026-06-30T13:00:00Z'));
    expect(nextCandleCloseMs(['H4'], now)).toBe(at('2026-06-30T16:00:00Z'));
  });

  it('returns the SOONEST boundary across several timeframes', () => {
    const now = at('2026-06-30T12:07:00Z');
    // M15 → 12:15, H1 → 13:00, H4 → 16:00 ⇒ soonest is 12:15
    expect(nextCandleCloseMs(['H4', 'H1', 'M15'], now)).toBe(
      at('2026-06-30T12:15:00Z'),
    );
  });

  it('always returns a STRICTLY future boundary, even exactly on a mark', () => {
    const onMark = at('2026-06-30T12:15:00Z');
    expect(nextCandleCloseMs(['M15'], onMark)).toBe(at('2026-06-30T12:30:00Z'));
  });

  it('returns null for an empty timeframe set', () => {
    expect(nextCandleCloseMs([], at('2026-06-30T12:07:00Z'))).toBeNull();
  });

  it('degrades unknown/weekly timeframes to a 15-minute cadence (never hangs)', () => {
    const now = at('2026-06-30T12:07:00Z');
    // W1 is broker-anchored, not epoch-aligned → falls back to a 15-min tick.
    expect(nextCandleCloseMs(['W1'], now)).toBe(at('2026-06-30T12:15:00Z'));
  });
});

describe('freshnessLabel', () => {
  const now = at('2026-06-30T12:03:30Z');
  it('says "à l\'instant" under a minute', () => {
    expect(freshnessLabel('2026-06-30T12:03:00Z', now)).toBe("à l'instant");
  });
  it('reflects the real age in minutes', () => {
    expect(freshnessLabel('2026-06-30T12:00:30Z', now)).toBe('il y a 3 min');
  });
  it('switches to hours past 60 minutes', () => {
    expect(freshnessLabel('2026-06-30T10:33:30Z', now)).toBe('il y a 2 h');
  });
  it('returns null for missing or unparseable timestamps', () => {
    expect(freshnessLabel(null, now)).toBeNull();
    expect(freshnessLabel('not-a-date', now)).toBeNull();
  });
});
