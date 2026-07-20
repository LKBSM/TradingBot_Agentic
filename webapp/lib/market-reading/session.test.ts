import { describe, expect, it } from 'vitest';
import {
  MARKET_STALE_THRESHOLD_SEC,
  isForexWeekend,
  isMarketClosed,
  isTwentyFourSevenMarket,
} from './session';

/** Epoch seconds for a UTC date string. */
const sec = (iso: string) => Math.floor(new Date(iso).getTime() / 1000);

describe('isTwentyFourSevenMarket', () => {
  it('flags crypto instruments as 24/7', () => {
    expect(isTwentyFourSevenMarket('BTCUSD')).toBe(true);
    expect(isTwentyFourSevenMarket('ETHUSD')).toBe(true);
  });
  it('does not flag FX / metals / indices', () => {
    for (const s of ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'US500']) {
      expect(isTwentyFourSevenMarket(s)).toBe(false);
    }
  });
});

describe('isForexWeekend', () => {
  it('is true across Saturday', () => {
    // 2026-05-23 is a Saturday.
    expect(isForexWeekend(new Date('2026-05-23T10:00:00Z'))).toBe(true);
  });
  it('is true Friday from 22:00 UTC, false just before', () => {
    expect(isForexWeekend(new Date('2026-05-22T22:00:00Z'))).toBe(true);
    expect(isForexWeekend(new Date('2026-05-22T21:59:00Z'))).toBe(false);
  });
  it('is true Sunday before 22:00 UTC, false at/after the reopen', () => {
    expect(isForexWeekend(new Date('2026-05-24T21:59:00Z'))).toBe(true);
    expect(isForexWeekend(new Date('2026-05-24T22:00:00Z'))).toBe(false);
  });
  it('is false on a normal weekday', () => {
    // 2026-05-20 is a Wednesday.
    expect(isForexWeekend(new Date('2026-05-20T12:00:00Z'))).toBe(false);
  });
});

describe('isMarketClosed', () => {
  const wednesday = new Date('2026-05-20T12:00:00Z'); // open FX session
  const saturday = new Date('2026-05-23T12:00:00Z'); // weekend

  it('reports crypto open even on the weekend', () => {
    expect(isMarketClosed('BTCUSD', { now: saturday, priceTs: null })).toBe(false);
  });

  it('reports FX closed on the weekend (calendar), even without a price', () => {
    expect(isMarketClosed('XAUUSD', { now: saturday, priceTs: null })).toBe(true);
    expect(isMarketClosed('EURUSD', { now: saturday, priceTs: null })).toBe(true);
  });

  it('reports FX open on a weekday when the feed is fresh', () => {
    const priceTs = sec('2026-05-20T11:55:00Z'); // 5 min old
    expect(isMarketClosed('XAUUSD', { now: wednesday, priceTs })).toBe(false);
  });

  it('a fresh feed overrides the weekend calendar (DST reopen edge)', () => {
    // Sunday 21:50 UTC is inside the calendar weekend, but a 2-min-old tick
    // proves the session already reopened — trust the data.
    const now = new Date('2026-05-24T21:50:00Z');
    const priceTs = sec('2026-05-24T21:48:00Z');
    expect(isMarketClosed('XAUUSD', { now, priceTs })).toBe(false);
  });

  it('reports closed on a weekday HOLIDAY via the staleness guard', () => {
    // Wednesday, but the last candle is a full day old → holiday-style silence.
    const priceTs = sec('2026-05-19T12:00:00Z'); // ~24 h old
    expect(isMarketClosed('XAUUSD', { now: wednesday, priceTs })).toBe(true);
  });

  it('does not flip on a sub-threshold weekday gap', () => {
    const priceTs = wednesday.getTime() / 1000 - (MARKET_STALE_THRESHOLD_SEC - 60);
    expect(isMarketClosed('XAUUSD', { now: wednesday, priceTs })).toBe(false);
  });
});
