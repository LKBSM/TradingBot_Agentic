import { describe, expect, it } from 'vitest';
import {
  countActiveZones,
  deriveTrendMaturity,
  formatBreakTimestamp,
  formatLastStructuralEvent,
  formatTrendMaturity,
  formatZoneDensity,
  timeframeMinutes,
} from '../regime-facts';
import type {
  BOSRecent,
  CHOCHRecent,
  FairValueGap,
  MarketReadingHeader,
  MarketReadingStructure,
  OrderBlock,
} from '@/types/market-reading';

const header = (
  overrides: Partial<MarketReadingHeader> = {},
): MarketReadingHeader => ({
  instrument: 'XAUUSD',
  timeframe: 'M15',
  candle_close_ts: '2026-06-24T19:00:00',
  close_price: 2400,
  ...overrides,
});

const choch = (overrides: Partial<CHOCHRecent> = {}): CHOCHRecent => ({
  direction: 'bullish',
  level: 2380,
  broken_at: '2026-06-24T14:30:00',
  validation_status: 'confirmed',
  ...overrides,
});

const bos = (overrides: Partial<BOSRecent> = {}): BOSRecent => ({
  direction: 'bearish',
  level: 2410,
  broken_at: '2026-06-24T18:00:00',
  validation_status: 'confirmed',
  ...overrides,
});

const ob = (status: OrderBlock['status'], id: string): OrderBlock => ({
  id,
  level_high: 2390,
  level_low: 2380,
  importance: 'medium',
  status,
  created_at: '2026-06-24T10:00:00',
  tested: false,
  user_flagged: false,
});

const fvg = (status: FairValueGap['status'], id: string): FairValueGap => ({
  id,
  level_high: 2395,
  level_low: 2390,
  status,
  created_at: '2026-06-24T10:00:00',
  tested: false,
  user_flagged: false,
});

const structure = (
  overrides: Partial<MarketReadingStructure> = {},
): MarketReadingStructure => ({
  bos: null,
  choch: null,
  order_blocks: [],
  fair_value_gaps: [],
  ...overrides,
});

describe('timeframeMinutes', () => {
  it('maps known timeframe codes to minutes', () => {
    expect(timeframeMinutes('M15')).toBe(15);
    expect(timeframeMinutes('H1')).toBe(60);
    expect(timeframeMinutes('H4')).toBe(240);
    expect(timeframeMinutes('D1')).toBe(1440);
  });
  it('returns null for an unknown code', () => {
    expect(timeframeMinutes('Z9')).toBeNull();
  });
});

describe('formatBreakTimestamp', () => {
  it('converts the engine UTC instant to the reader timezone (pinned UTC here)', () => {
    // Naive engine timestamps are UTC → shown as-is in the UTC zone.
    expect(formatBreakTimestamp('2026-06-24T14:30:00', 'UTC')).toBe('24/06 à 14:30');
    // A +02:00 instant is 07:05 UTC.
    expect(formatBreakTimestamp('2026-01-02T09:05:00+02:00', 'UTC')).toBe('02/01 à 07:05');
  });
  it('returns null on an unparseable string', () => {
    expect(formatBreakTimestamp('not-a-date')).toBeNull();
  });
});

describe('deriveTrendMaturity (b)', () => {
  it('derives the candle count since the point-in-time CHOCH (fallback)', () => {
    const m = deriveTrendMaturity(structure({ choch: choch() }), header());
    // 14:30 → 19:00 = 270 min / 15 = 18 candles.
    expect(m).toEqual({
      direction: 'bullish',
      brokenAt: '2026-06-24T14:30:00',
      bars: 18,
    });
  });
  it('anchors on the MOST RECENT CHOCH from the event history (even bars ago)', () => {
    // The point-in-time choch is null, but a CHOCH lives in the history 20 bars
    // back → maturity must use it (not « non disponible », never a BOS).
    const m = deriveTrendMaturity(
      structure({
        choch: null,
        bos: bos(), // a BOS present must be IGNORED for maturity
        choch_events: [
          choch({ broken_at: '2026-06-24T10:00:00', direction: 'bearish' }),
          choch({ broken_at: '2026-06-24T14:00:00', direction: 'bearish' }),
        ],
      }),
      header(),
    );
    // Latest event = 14:00 → 19:00 = 300 min / 15 = 20 candles.
    expect(m).toEqual({
      direction: 'bearish',
      brokenAt: '2026-06-24T14:00:00',
      bars: 20,
    });
  });
  it('never uses a BOS for maturity — BOS-only → null', () => {
    expect(deriveTrendMaturity(structure({ bos: bos() }), header())).toBeNull();
  });
  it('returns null when no CHOCH exists anywhere in the window', () => {
    expect(deriveTrendMaturity(structure(), header())).toBeNull();
  });
  it('leaves bars null for an unknown timeframe (no invented count)', () => {
    const m = deriveTrendMaturity(structure({ choch: choch() }), header({ timeframe: 'Z9' }));
    expect(m?.bars).toBeNull();
  });
  it('guards against a future broken_at (negative diff → bars null)', () => {
    const m = deriveTrendMaturity(
      structure({ choch: choch({ broken_at: '2026-06-25T00:00:00' }) }),
      header(),
    );
    expect(m?.bars).toBeNull();
  });
});

describe('formatTrendMaturity (b)', () => {
  it('present-tense line with date + derived candle count', () => {
    expect(formatTrendMaturity(structure({ choch: choch() }), header(), 'UTC')).toBe(
      'Structure orientée haussière depuis le CHOCH du 24/06 à 14:30 (≈ 18 bougies M15).',
    );
  });
  it('bearish orientation', () => {
    expect(
      formatTrendMaturity(structure({ choch: choch({ direction: 'bearish' }) }), header(), 'UTC'),
    ).toBe('Structure orientée baissière depuis le CHOCH du 24/06 à 14:30 (≈ 18 bougies M15).');
  });
  it('omits the candle count when the timeframe is unknown', () => {
    expect(
      formatTrendMaturity(structure({ choch: choch() }), header({ timeframe: 'Z9' }), 'UTC'),
    ).toBe('Structure orientée haussière depuis le CHOCH du 24/06 à 14:30.');
  });
  it('uses the most recent CHOCH from the history for the line', () => {
    expect(
      formatTrendMaturity(
        structure({
          choch: null,
          choch_events: [choch({ broken_at: '2026-06-24T14:00:00', direction: 'bearish' })],
        }),
        header(),
        'UTC',
      ),
    ).toBe('Structure orientée baissière depuis le CHOCH du 24/06 à 14:00 (≈ 20 bougies M15).');
  });
  it('returns null when no CHOCH (BOS-only or empty → « non disponible »)', () => {
    expect(formatTrendMaturity(structure({ bos: bos() }), header())).toBeNull();
    expect(formatTrendMaturity(structure(), header())).toBeNull();
  });
});

describe('formatLastStructuralEvent (c)', () => {
  it('phrases the CHOCH event with TF', () => {
    expect(
      formatLastStructuralEvent(
        structure({ choch: choch({ direction: 'bearish' }) }),
        header({ timeframe: 'H1' }),
      ),
    ).toBe('CHOCH baissier confirmé (H1)');
  });
  it('picks the most recent of BOS / CHOCH by broken_at', () => {
    // bos at 18:00 is later than choch at 14:30 → BOS surfaces.
    expect(
      formatLastStructuralEvent(structure({ bos: bos(), choch: choch() }), header()),
    ).toBe('BOS baissier confirmé (M15)');
  });
  it('falls back to the only present break', () => {
    expect(formatLastStructuralEvent(structure({ bos: bos() }), header())).toBe(
      'BOS baissier confirmé (M15)',
    );
  });
  it('renders pending / invalidated validation states', () => {
    expect(
      formatLastStructuralEvent(
        structure({ choch: choch({ validation_status: 'pending' }) }),
        header(),
      ),
    ).toBe('CHOCH haussier en attente de confirmation (M15)');
  });
  it('returns null when neither BOS nor CHOCH (caller → « non disponible »)', () => {
    expect(formatLastStructuralEvent(structure(), header())).toBeNull();
  });
});

describe('countActiveZones / formatZoneDensity (d)', () => {
  it('counts only status === active', () => {
    const s = structure({
      order_blocks: [ob('active', 'a'), ob('mitigated', 'b'), ob('active', 'c')],
      fair_value_gaps: [fvg('active', 'd'), fvg('filled', 'e')],
    });
    expect(countActiveZones(s)).toEqual({ ob: 2, fvg: 1 });
    expect(formatZoneDensity(s)).toBe('2 OB · 1 FVG actifs');
  });
  it('0 · 0 is a real fact, not « non disponible »', () => {
    expect(formatZoneDensity(structure())).toBe('0 OB · 0 FVG actifs');
  });
});

describe('no predictive / probabilistic vocabulary', () => {
  it('every produced line stays strictly descriptive', () => {
    const lines = [
      formatTrendMaturity(structure({ choch: choch() }), header()),
      formatLastStructuralEvent(structure({ bos: bos(), choch: choch() }), header()),
      formatZoneDensity(
        structure({ order_blocks: [ob('active', 'a')], fair_value_gaps: [fvg('active', 'b')] }),
      ),
    ].join(' ');
    const forbidden = [
      /probab/i,
      /\d+\s*%/,
      /va\s/i,
      /devrait/i,
      /attendre/i,
      /signal/i,
      /objectif/i,
      /surveiller/i,
      /risqu/i,
    ];
    for (const re of forbidden) expect(lines).not.toMatch(re);
  });
});
