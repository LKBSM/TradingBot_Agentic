import { describe, expect, it } from 'vitest';
import { formationSession } from '../formation-session';

// America/New_York wall-clock windows; the timestamps below are UTC (EDT = −4 in May).
describe('formationSession — mirrors the backend NY session windows', () => {
  it('New York session (08:00–17:00 NY) — 12:00Z = 08:00 NY', () => {
    expect(formationSession('2026-05-26T12:00:00Z', 'XAUUSD')).toBe('new_york');
  });
  it('London before the NY overlap — 09:00Z = 05:00 NY', () => {
    expect(formationSession('2026-05-26T09:00:00Z', 'XAUUSD')).toBe('london');
  });
  it('Asia session — 23:00Z = 19:00 NY (wraps past midnight)', () => {
    expect(formationSession('2026-05-26T23:00:00Z', 'XAUUSD')).toBe('asia');
  });
  it('crypto has no session découpage → null', () => {
    expect(formationSession('2026-05-26T12:00:00Z', 'BTCUSD')).toBeNull();
  });
  it('unparsable timestamp → null (never fabricated)', () => {
    expect(formationSession('not-a-date', 'EURUSD')).toBeNull();
  });
});
