import { describe, expect, it } from 'vitest';
import {
  formatLocalDayHm,
  formatLocalHm,
  parseUtc,
  utcOffsetLabel,
} from '../localTime';

describe('parseUtc', () => {
  it('treats a naive engine timestamp as UTC', () => {
    const d = parseUtc('2026-06-24T14:30:00');
    expect(d?.toISOString()).toBe('2026-06-24T14:30:00.000Z');
  });
  it('respects an explicit offset', () => {
    const d = parseUtc('2026-01-02T09:05:00+02:00');
    expect(d?.toISOString()).toBe('2026-01-02T07:05:00.000Z');
  });
  it('accepts a trailing Z', () => {
    expect(parseUtc('2026-06-24T14:30:00Z')?.toISOString()).toBe(
      '2026-06-24T14:30:00.000Z',
    );
  });
  it('returns null on garbage / empty', () => {
    expect(parseUtc('not-a-date')).toBeNull();
    expect(parseUtc('')).toBeNull();
    expect(parseUtc(null)).toBeNull();
  });
});

describe('formatLocalHm / formatLocalDayHm (pinned UTC, locale-aware)', () => {
  const d = parseUtc('2026-06-24T14:30:00')!;
  it('formats HH:MM in the given locale + zone', () => {
    expect(formatLocalHm(d, 'fr', 'UTC')).toBe('14:30');
    expect(formatLocalHm(d, 'en', 'UTC')).toBe('14:30');
  });
  it('formats an UNAMBIGUOUS named-month date + time per locale (never day/month digits)', () => {
    // A named month removes the en day/month ambiguity a numeric « 24/06 » carries.
    expect(formatLocalDayHm(d, 'fr', 'UTC')).toBe('24 juin · 14:30');
    // en orders month-first with a short name; 24-hour clock (never 02:30 PM).
    expect(formatLocalDayHm(d, 'en', 'UTC')).toBe('Jun 24 · 14:30');
    // es keeps day-first with a short Spanish month.
    expect(formatLocalDayHm(d, 'es', 'UTC')).toBe('24 jun · 14:30');
  });
});

describe('utcOffsetLabel', () => {
  it('labels UTC when the offset is zero', () => {
    expect(utcOffsetLabel(0)).toBe('UTC');
  });
  it('labels a negative offset with a real minus sign', () => {
    expect(utcOffsetLabel(-240)).toBe('UTC−4');
  });
  it('labels a positive offset', () => {
    expect(utcOffsetLabel(120)).toBe('UTC+2');
  });
  it('includes minutes when the offset is not whole hours', () => {
    expect(utcOffsetLabel(-210)).toBe('UTC−3:30');
  });
});
