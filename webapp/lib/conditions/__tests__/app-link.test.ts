import { describe, expect, it } from 'vitest';
import { buildAppHref, resolveComboFromQuery } from '../app-link';

describe('buildAppHref (Analyser deep-link)', () => {
  it('points at /app and carries the exact instrument + timeframe', () => {
    const href = buildAppHref('fr', { instrument: 'XAUUSD', timeframe: 'H1' });
    expect(href).toContain('/app');
    expect(href).toContain('instrument=XAUUSD');
    expect(href).toContain('timeframe=H1');
  });

  it('omits the prefix for the default locale, prefixes others', () => {
    expect(buildAppHref('fr', { instrument: 'EURUSD', timeframe: 'M15' })).toBe(
      '/app?instrument=EURUSD&timeframe=M15',
    );
    expect(buildAppHref('en', { instrument: 'EURUSD', timeframe: 'M15' })).toBe(
      '/en/app?instrument=EURUSD&timeframe=M15',
    );
  });
});

describe('resolveComboFromQuery', () => {
  it('accepts a valid in-perimeter combo', () => {
    expect(resolveComboFromQuery('XAUUSD', 'H4')).toEqual({
      instrument: 'XAUUSD',
      timeframe: 'H4',
    });
  });

  it('rejects out-of-perimeter or missing values', () => {
    expect(resolveComboFromQuery('BTCUSD', 'H1')).toBeNull();
    expect(resolveComboFromQuery('XAUUSD', 'M30')).toBeNull();
    expect(resolveComboFromQuery(undefined, 'H1')).toBeNull();
    expect(resolveComboFromQuery('XAUUSD', undefined)).toBeNull();
  });
});
