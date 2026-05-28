import { describe, expect, it } from 'vitest';
import {
  convictionLabelLong,
  convictionTone,
  directionBadgeVariant,
  formatChangepointStability,
  formatHitRate,
  formatHmmLabel,
  formatInstrument,
  formatJumpDescriptor,
  formatNextEventCountdown,
  formatPips,
  formatPipsRange,
  formatPrice,
  formatProfitFactor,
  formatRegimeGate,
  formatRelativePast,
  formatRetestState,
  formatSession,
  formatSignedPercent,
  formatTimeframe,
  formatValidityCountdown,
  formatVerdict,
  formatVolatilityRegime,
  formatZone,
} from './insight-formatters';
import type { InsightSignalV2 } from '@/types/insight';

function makeSignal(overrides: Partial<InsightSignalV2> = {}): InsightSignalV2 {
  // Minimal valid signal — enough for the formatters under test.
  return {
    schema_version: '2.1.0',
    id: 'test-id',
    instrument: 'XAUUSD',
    timeframe: 'M15',
    created_at_utc: '2026-05-26T11:47:00+00:00',
    valid_until_utc: '2026-05-26T15:47:00+00:00',
    direction: 'BULLISH_SETUP',
    conviction_0_100: 72,
    conviction_label: 'strong',
    uncertainty: {
      conformal_lower: 54,
      conformal_upper: 82,
      coverage_alpha: 0.1,
      n_calibration: 2000,
      empirical_coverage: 0.91,
    },
    structure_readout: {
      bos_level: 2391.5,
      bos_event_age_bars: 2,
      choch_present: true,
      fvg_zone: [2378, 2381],
      fvg_size_atr: 0.42,
      ob_zone: [2375, 2378],
      ob_strength: 0.73,
      retest_state: 'armed',
      structural_invalidation: 2378,
      liquidity_zone_upper: null,
      liquidity_zone_lower: null,
    },
    regime_readout: {
      hmm_label: 'trend_bullish',
      hmm_posterior: 0.71,
      bocpd_changepoint_prob: 0.03,
      expected_run_length: 180,
      jump_ratio: 0.12,
      regime_gate_decision: 'TRADE',
    },
    volatility_readout: {
      regime: 'normal',
      forecast_atr_pips: 8.7,
      naive_atr_pips: 7.9,
      forecast_vs_naive_pct: 10.13,
      confidence_interval_pips: [7.2, 10.4],
      is_fallback: false,
    },
    event_readout: {
      news_blackout_active: false,
      next_event_label: 'FOMC Minutes',
      next_event_in_minutes: 1083,
      sentiment_score: 0.3,
      sentiment_confidence: 0.7,
      session: 'ny_overlap',
    },
    breakdown_components: [],
    historical_stats: null,
    narrative_short: '',
    narrative_long: null,
    narrative_language: 'fr',
    sources_cited: [],
    compliance: {
      disclaimer_lang: 'fr',
      jurisdiction_blocked: [],
      edge_claim: false,
      is_paper_demo: true,
    },
    ...overrides,
  };
}

describe('formatInstrument / formatTimeframe', () => {
  it('returns French human labels for known instruments', () => {
    expect(formatInstrument({ instrument: 'XAUUSD' })).toBe('Or (XAU/USD)');
    expect(formatInstrument({ instrument: 'EURUSD' })).toBe('Euro / Dollar (EUR/USD)');
    expect(formatInstrument({ instrument: 'USDJPY' })).toBe('Dollar / Yen (USD/JPY)');
  });

  it('returns timeframe labels', () => {
    expect(formatTimeframe({ timeframe: 'M15' })).toBe('15 minutes');
    expect(formatTimeframe({ timeframe: 'H4' })).toBe('4 heures');
    expect(formatTimeframe({ timeframe: 'D1' })).toBe('1 jour');
  });
});

describe('formatVerdict', () => {
  it('produces a French sentence for bullish XAU', () => {
    const s = makeSignal({ direction: 'BULLISH_SETUP', conviction_label: 'strong' });
    expect(formatVerdict(s)).toBe(`Lecture haussière sur l'or, conviction marquée.`);
  });

  it('produces a French sentence for bearish EUR', () => {
    const s = makeSignal({
      instrument: 'EURUSD',
      direction: 'BEARISH_SETUP',
      conviction_label: 'moderate',
    });
    expect(formatVerdict(s)).toBe(`Lecture baissière sur l'euro, conviction modérée.`);
  });

  it('produces a neutral sentence', () => {
    const s = makeSignal({ direction: 'NEUTRAL', conviction_label: 'weak' });
    expect(formatVerdict(s)).toBe(`Lecture neutre sur l'or, conviction faible.`);
  });

  it('NEVER contains a prescriptive verb (compliance check)', () => {
    const forbidden = [/\bachet/i, /\bvend/i, /\bsignal d/i, /\bopportunité/i, /\bgarant/i];
    for (const direction of ['BULLISH_SETUP', 'BEARISH_SETUP', 'NEUTRAL'] as const) {
      for (const label of ['weak', 'moderate', 'strong', 'institutional'] as const) {
        const s = makeSignal({ direction, conviction_label: label });
        const verdict = formatVerdict(s);
        for (const token of forbidden) {
          expect(verdict).not.toMatch(token);
        }
      }
    }
  });
});

describe('conviction labels', () => {
  it('tone maps to FR adjectives', () => {
    expect(convictionTone('weak')).toBe('faible');
    expect(convictionTone('moderate')).toBe('modérée');
    expect(convictionTone('strong')).toBe('marquée');
    expect(convictionTone('institutional')).toBe('institutionnelle');
  });

  it('long label maps to full phrase', () => {
    expect(convictionLabelLong('strong')).toBe('Conviction marquée');
  });
});

describe('directionBadgeVariant', () => {
  it('routes bullish→bull, bearish→bear, neutral→neutral', () => {
    expect(directionBadgeVariant('BULLISH_SETUP')).toBe('bull');
    expect(directionBadgeVariant('BEARISH_SETUP')).toBe('bear');
    expect(directionBadgeVariant('NEUTRAL')).toBe('neutral');
  });
});

describe('formatRelativePast', () => {
  const NOW = new Date('2026-05-26T12:00:00Z');

  it(`returns "à l'instant" under 1 minute`, () => {
    expect(formatRelativePast('2026-05-26T11:59:30Z', NOW)).toBe(`à l'instant`);
  });

  it('returns minutes', () => {
    expect(formatRelativePast('2026-05-26T11:55:00Z', NOW)).toBe('il y a 5 minutes');
    expect(formatRelativePast('2026-05-26T11:59:00Z', NOW)).toBe('il y a 1 minute');
  });

  it('returns hours', () => {
    expect(formatRelativePast('2026-05-26T08:00:00Z', NOW)).toBe('il y a 4 heures');
    expect(formatRelativePast('2026-05-26T11:00:00Z', NOW)).toBe('il y a 1 heure');
  });

  it('returns days', () => {
    expect(formatRelativePast('2026-05-23T12:00:00Z', NOW)).toBe('il y a 3 jours');
  });
});

describe('formatValidityCountdown', () => {
  const NOW = new Date('2026-05-26T12:00:00Z');

  it('handles future expirations within an hour', () => {
    const out = formatValidityCountdown('2026-05-26T12:45:00Z', NOW);
    expect(out.expired).toBe(false);
    expect(out.label).toBe('expire dans 45 min');
  });

  it('handles future expirations hours+minutes', () => {
    const out = formatValidityCountdown('2026-05-26T14:47:00Z', NOW);
    expect(out.expired).toBe(false);
    expect(out.label).toBe('expire dans 2h47');
  });

  it('flags expired signals', () => {
    const out = formatValidityCountdown('2026-05-26T10:00:00Z', NOW);
    expect(out.expired).toBe(true);
    expect(out.label).toMatch(/expirée il y a 2 heures/);
  });
});

describe('formatNextEventCountdown', () => {
  it('returns null when no event', () => {
    expect(formatNextEventCountdown(null)).toBeNull();
  });
  it('formats minutes < 60', () => {
    expect(formatNextEventCountdown(42)).toBe('dans 42 min');
  });
  it('formats hours+minutes', () => {
    expect(formatNextEventCountdown(180)).toBe('dans 3h');
    expect(formatNextEventCountdown(167)).toBe('dans 2h47');
  });
  it('formats days', () => {
    expect(formatNextEventCountdown(60 * 24 * 2)).toBe('dans 2j');
    expect(formatNextEventCountdown(60 * 24 * 2 + 60 * 5)).toBe('dans 2j5h');
  });
});

describe('price + zone formatting', () => {
  it('uses 2 decimals for XAU', () => {
    expect(formatPrice(2391.5, 'XAUUSD')).toMatch(/2.{1,3}391,50/);
  });

  it('uses 5 decimals for EUR', () => {
    expect(formatPrice(1.0875, 'EURUSD')).toBe('1,08750');
  });

  it('uses 3 decimals for JPY', () => {
    expect(formatPrice(150.123, 'USDJPY')).toBe('150,123');
  });

  it('handles null zone', () => {
    expect(formatZone(null, 'XAUUSD')).toBeNull();
  });

  it('formats zone with the instrument precision', () => {
    expect(formatZone([2378, 2381], 'XAUUSD')).toMatch(/2.{1,3}378,00 – 2.{1,3}381,00/);
  });
});

describe('retest state labels', () => {
  it.each([
    ['idle', 'aucun retest en cours'],
    ['awaiting', 'en attente de retest'],
    ['armed', 'retest armé'],
    ['consumed', 'retest dépassé'],
  ] as const)('%s → %s', (state, expected) => {
    expect(formatRetestState(state)).toBe(expected);
  });
});

describe('regime helpers', () => {
  it('maps HMM labels to FR', () => {
    expect(formatHmmLabel('trend_bullish')).toBe('Tendance haussière');
    expect(formatHmmLabel('range_low_vol')).toBe('Consolidation calme');
    expect(formatHmmLabel('high_vol_stress')).toBe('Stress / forte volatilité');
  });

  it('maps gate decisions to tone+label', () => {
    expect(formatRegimeGate('TRADE')).toEqual({
      label: 'Conditions favorables',
      tone: 'ok',
    });
    expect(formatRegimeGate('REDUCE').tone).toBe('warn');
    expect(formatRegimeGate('BLOCK').tone).toBe('block');
  });

  it('changepoint stability thresholds', () => {
    expect(formatChangepointStability(0.02).tone).toBe('ok');
    expect(formatChangepointStability(0.06).tone).toBe('warn');
    expect(formatChangepointStability(0.12).tone).toBe('block');
  });

  it('jump descriptor thresholds', () => {
    expect(formatJumpDescriptor(0.1).tone).toBe('ok');
    expect(formatJumpDescriptor(0.3).tone).toBe('warn');
    expect(formatJumpDescriptor(0.5).tone).toBe('block');
  });
});

describe('volatility helpers', () => {
  it('regime → FR label', () => {
    expect(formatVolatilityRegime('low')).toBe('Volatilité basse');
    expect(formatVolatilityRegime('high')).toBe('Volatilité élevée');
  });

  it('formatPips uses FR decimal comma', () => {
    expect(formatPips(8.7)).toBe('8,7 pips');
  });

  it('formatPipsRange', () => {
    expect(formatPipsRange([7.2, 10.4])).toBe('7,2 pips – 10,4 pips');
  });

  it('formatSignedPercent uses minus sign Unicode', () => {
    expect(formatSignedPercent(10.13)).toBe('+10,1 %');
    expect(formatSignedPercent(-5.4)).toBe('−5,4 %');
  });
});

describe('session + historical', () => {
  it('session labels', () => {
    expect(formatSession('ny_overlap')).toBe('Overlap Londres / New York');
    expect(formatSession('asian')).toBe('Session asiatique');
  });

  it('profit factor formats FR comma with 2 decimals', () => {
    // 1.30 (not 1.31) because (1.305).toFixed(2) === '1.30' in V8 due to
    // IEEE 754 representation of 1.305 ≈ 1.30499999…
    expect(formatProfitFactor(1.305)).toBe('1,30');
    expect(formatProfitFactor(1.31)).toBe('1,31');
    expect(formatProfitFactor(0.95)).toBe('0,95');
  });

  it('hit rate formats integer percent', () => {
    expect(formatHitRate(0.319)).toBe('32 %');
    expect(formatHitRate(0.5)).toBe('50 %');
  });
});
