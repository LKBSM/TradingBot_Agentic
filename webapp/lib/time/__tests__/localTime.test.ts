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

describe('formatLocalHm / formatLocalDayHm (pinned UTC)', () => {
  const d = parseUtc('2026-06-24T14:30:00')!;
  it('formats HH:MM in the given zone', () => {
    expect(formatLocalHm(d, 'UTC')).toBe('14:30');
  });
  it('formats JJ/MM à HH:MM in the given zone', () => {
    expect(formatLocalDayHm(d, 'UTC')).toBe('24/06 à 14:30');
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
