import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';
import {
  deriveMarketStatus,
  badgeLabelKey,
  badgeTitleKey,
  formatNyTimestamp,
} from '@/lib/market-reading/status';
import type { MarketStatusPayload } from '@/types/market-reading';

function payload(state: MarketStatusPayload['state']): MarketStatusPayload {
  return {
    state,
    reason: state,
    instrument: 'XAUUSD',
    timeframe: 'M15',
    last_close_ts: '2026-07-24T21:00:00Z',
    next_open_ts: state === 'open' || state === 'data_lagged' ? null : '2026-07-26T22:00:00Z',
    bars_behind: state === 'data_lagged' ? 12 : null,
  };
}

describe('MC-1 deriveMarketStatus', () => {
  it('returns null when no server payload', () => {
    expect(deriveMarketStatus(null)).toBeNull();
    expect(deriveMarketStatus(undefined)).toBeNull();
  });

  it('closed_weekend / closed_holiday / daily_break are closed and not live', () => {
    for (const s of ['closed_weekend', 'closed_holiday', 'daily_break'] as const) {
      const v = deriveMarketStatus(payload(s))!;
      expect(v.isClosed).toBe(true);
      expect(v.isLive).toBe(false);
      expect(v.isLagged).toBe(false);
    }
  });

  it('data_lagged is lagged but not "closed"', () => {
    const v = deriveMarketStatus(payload('data_lagged'))!;
    expect(v.isLagged).toBe(true);
    expect(v.isClosed).toBe(false);
    expect(v.isLive).toBe(false);
  });

  it('open is live', () => {
    const v = deriveMarketStatus(payload('open'))!;
    expect(v.isLive).toBe(true);
    expect(v.isClosed).toBe(false);
  });
});

describe('MC-1 badge keys', () => {
  it('maps each state to a chart.* label / title key (open has none)', () => {
    expect(badgeLabelKey('closed_weekend')).toBe('chart.marketClosed');
    expect(badgeLabelKey('daily_break')).toBe('chart.marketPaused');
    expect(badgeLabelKey('data_lagged')).toBe('chart.dataLagged');
    expect(badgeLabelKey('open')).toBeNull();
    expect(badgeTitleKey('daily_break')).toBe('chart.marketPausedTitle');
    expect(badgeTitleKey('open')).toBeNull();
  });
});

describe('MC-1 formatNyTimestamp', () => {
  it('formats a UTC instant in New-York wall clock', () => {
    // 2026-07-24T21:00Z = Friday 17:00 New York (EDT).
    const s = formatNyTimestamp('2026-07-24T21:00:00Z', 'en')!;
    expect(s.toLowerCase()).toContain('friday');
    expect(s).toContain('17:00');
  });

  it('returns null for missing / invalid input', () => {
    expect(formatNyTimestamp(null, 'en')).toBeNull();
    expect(formatNyTimestamp('not-a-date', 'en')).toBeNull();
  });
});

// --------------------------------------------------------------------------- //
// Copy honesty — the new MC-1 strings state present facts, never a forecast.
// --------------------------------------------------------------------------- //
const MC1_CHART_KEYS = [
  'marketClosed',
  'marketPaused',
  'dataLagged',
  'lastCandleClosed',
  'reopensAt',
  'noNewCandleSince',
] as const;

// Predictive / directional language that must never appear in a closed-market
// message (fr + en). We assert absence case-insensitively.
const FORBIDDEN = [
  'rebond',
  'opportun',
  'se prépare',
  'prepares',
  'en attente',
  'devrait',
  'should ',
  'va chercher',
  'prochain mouvement',
  'next move',
];

describe('MC-1 copy honesty', () => {
  for (const [name, bundle] of [
    ['fr', fr],
    ['en', en],
  ] as const) {
    it(`${name}: every MC-1 chart key resolves to a non-empty string`, () => {
      const chart = (bundle as Record<string, any>).app.chart as Record<string, string>;
      for (const k of MC1_CHART_KEYS) {
        expect(typeof chart[k], `${name}.app.chart.${k}`).toBe('string');
        expect((chart[k] ?? '').length).toBeGreaterThan(0);
      }
      expect(typeof (bundle as Record<string, any>).scanner.noNewClose).toBe('string');
    });

    it(`${name}: no MC-1 string makes a predictive/directional promise`, () => {
      const chart = (bundle as Record<string, any>).app.chart as Record<string, string>;
      const strings = [
        ...MC1_CHART_KEYS.map((k) => chart[k]),
        chart.marketClosedTitle,
        chart.marketPausedTitle,
        chart.dataLaggedTitle,
        (bundle as Record<string, any>).scanner.noNewClose,
      ];
      for (const s of strings) {
        const lower = s.toLowerCase();
        for (const bad of FORBIDDEN) {
          expect(lower.includes(bad), `"${s}" must not contain « ${bad} »`).toBe(false);
        }
      }
    });
  }
});
