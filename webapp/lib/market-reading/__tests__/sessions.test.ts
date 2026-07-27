import { describe, it, expect } from 'vitest';
import { computeSession, formatWeeklyClose, splitDelay } from '../sessions';
import type { MarketStatusPayload } from '@/types/market-reading';

// Standard FX/metal windows (ET), as the server publishes them.
const base: MarketStatusPayload = {
  state: 'open',
  reason: 'open',
  instrument: 'XAUUSD',
  timeframe: 'H1',
  last_close_ts: null,
  next_open_ts: null,
  bars_behind: 0,
  continuous: false,
  session_tz: 'America/New_York',
  sessions: [
    { name: 'asia', start: '19:00', end: '04:00' },
    { name: 'london', start: '03:00', end: '11:30' },
    { name: 'new_york', start: '08:00', end: '17:00' },
  ],
  weekly_close: { weekday: 4, time: '17:00' },
};

// July = EDT (UTC−4). A UTC instant of HH:00Z maps to NY (HH−4):00.
const nyJuly = (utcHour: number, utcMin = 0) =>
  new Date(Date.UTC(2026, 6, 22, utcHour, utcMin, 0)); // Wed 22 Jul 2026

describe('computeSession — current session by NY wall-clock', () => {
  it('mid-New-York afternoon → New York, next transition is the close', () => {
    const s = computeSession(base, nyJuly(19, 20))!; // 15:20 ET
    expect(s.current).toBe('new_york');
    expect(s.localTime).toBe('15:20');
    expect(s.next).toEqual({ label: 'close', at: '17:00', inMinutes: 100 });
    expect(s.overlap?.state).toBe('ended');
    expect(s.currentRange).toEqual({ start: '08:00', end: '17:00' });
  });

  it('London/NY overlap → overlap, overlap state ongoing', () => {
    const s = computeSession(base, nyJuly(13, 0))!; // 09:00 ET
    expect(s.current).toBe('overlap');
    expect(s.overlap?.state).toBe('ongoing');
    expect(s.currentRange).toEqual({ start: '08:00', end: '11:30' });
  });

  it('early morning ET → London, next is New York opening', () => {
    const s = computeSession(base, nyJuly(9, 0))!; // 05:00 ET
    expect(s.current).toBe('london');
    expect(s.next?.label).toBe('new_york');
    expect(s.next?.at).toBe('08:00');
    expect(s.overlap?.state).toBe('upcoming');
  });

  it('between NY close and Asia open → outside, next is Asia', () => {
    const s = computeSession(base, nyJuly(22, 0))!; // 18:00 ET
    expect(s.current).toBe('outside');
    expect(s.next).toEqual({ label: 'asia', at: '19:00', inMinutes: 60 });
  });
});

describe('computeSession — continuous & closed', () => {
  it('continuous market → « continuous », no windows/close', () => {
    const s = computeSession({ ...base, continuous: true, sessions: [] }, nyJuly(13))!;
    expect(s.current).toBe('continuous');
    expect(s.next).toBeNull();
    expect(s.overlap).toBeNull();
  });

  it('closed market → outside + next open transition', () => {
    const s = computeSession(
      { ...base, state: 'closed_weekend', next_open_ts: '2026-07-26T21:00:00Z' },
      new Date(Date.UTC(2026, 6, 25, 21, 0)),
    )!;
    expect(s.current).toBe('outside');
    expect(s.next?.label).toBe('open');
    expect(s.next?.inMinutes).toBeGreaterThan(0);
  });

  it('no session data → null', () => {
    expect(computeSession(null, nyJuly(13))).toBeNull();
    expect(computeSession({ ...base, sessions: [] }, nyJuly(13))).toBeNull();
  });
});

describe('helpers', () => {
  it('splitDelay decomposes minutes into d/h/m', () => {
    expect(splitDelay(100)).toEqual({ d: 0, h: 1, m: 40 });
    expect(splitDelay(45)).toEqual({ d: 0, h: 0, m: 45 });
    expect(splitDelay(1755)).toEqual({ d: 1, h: 5, m: 15 });
  });

  it('formatWeeklyClose localizes the weekday from the server weekday index', () => {
    expect(formatWeeklyClose({ weekday: 4, time: '17:00' }, 'en')).toMatch(/Fri 17:00/);
    expect(formatWeeklyClose(null, 'en')).toBeNull();
  });
});
