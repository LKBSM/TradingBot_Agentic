import { describe, expect, it } from 'vitest';
import { suggestedQuestions } from './suggested-questions';
import type { InsightSignalV2 } from '@/types/insight';

function makeSignal(overrides: Partial<InsightSignalV2> = {}): InsightSignalV2 {
  // Minimal-but-valid signal — only fields the suggested-question logic reads.
  const base = {
    schema_version: '2.1.0',
    id: 'test',
    instrument: 'XAUUSD',
    timeframe: 'M15',
    created_at_utc: '2026-05-28T00:00:00Z',
    valid_until_utc: '2026-05-28T01:00:00Z',
    direction: 'BULLISH_SETUP',
    conviction_0_100: 65,
    conviction_label: 'moderate',
    uncertainty: {
      conformal_lower: 54,
      conformal_upper: 82,
      coverage_alpha: 0.1,
      n_calibration: 200,
      empirical_coverage: 0.91,
    },
    structure_readout: {
      bos_level: 2350,
      bos_event_age_bars: 3,
      choch_present: true,
      fvg_zone: null,
      fvg_size_atr: null,
      ob_zone: null,
      ob_strength: null,
      retest_state: 'armed',
      structural_invalidation: 2340,
      liquidity_zone_upper: null,
      liquidity_zone_lower: null,
    },
    regime_readout: {
      hmm_label: 'trend_bullish',
      hmm_posterior: 0.7,
      bocpd_changepoint_prob: 0.05,
      expected_run_length: 24,
      jump_ratio: 0.1,
      regime_gate_decision: 'TRADE',
    },
    volatility_readout: {
      regime: 'normal',
      forecast_atr_pips: 12,
      naive_atr_pips: 14,
      forecast_vs_naive_pct: -14,
      confidence_interval_pips: [10, 14],
      is_fallback: false,
    },
    event_readout: {
      news_blackout_active: false,
      next_event_label: null,
      next_event_in_minutes: null,
      sentiment_score: 0,
      sentiment_confidence: 0,
      session: 'EUROPE',
    },
    breakdown_components: [
      { name: 'bos', contribution: 9, weight_max: 15, reasoning: 'BOS fresh' },
      { name: 'order_block', contribution: 12, weight_max: 15, reasoning: 'OB strong' },
      { name: 'fvg', contribution: 5, weight_max: 10, reasoning: 'FVG present' },
    ],
    historical_stats: null,
    narrative_short: 'test',
    narrative_long: null,
  } as unknown as InsightSignalV2;
  return { ...base, ...overrides };
}

describe('DG-114 suggested questions — Q1 conviction tier', () => {
  it('weak → "pourquoi aussi basse"', () => {
    const s = makeSignal({ conviction_label: 'weak' });
    const [q1] = suggestedQuestions(s, 'fr');
    expect(q1.source).toBe('q1_conviction');
    expect(q1.text.toLowerCase()).toContain('basse');
  });

  it('institutional → "pourquoi atteint-il le label"', () => {
    const s = makeSignal({ conviction_label: 'institutional' });
    const [q1] = suggestedQuestions(s, 'fr');
    expect(q1.text.toLowerCase()).toContain('institutional');
  });

  it('strong (EN) renders English copy', () => {
    const s = makeSignal({ conviction_label: 'strong' });
    const [q1] = suggestedQuestions(s, 'en');
    expect(q1.text).toContain('strong conviction');
  });
});

describe('DG-114 suggested questions — Q2 top component', () => {
  it('picks the top-contributing component', () => {
    const s = makeSignal();
    const [, q2] = suggestedQuestions(s, 'fr');
    expect(q2.source).toBe('q2_top_component');
    // top is order_block (contribution 12)
    expect(q2.text.toLowerCase()).toContain('order block');
  });

  it('falls back to "8 composantes" when components are empty', () => {
    const s = makeSignal({ breakdown_components: [] });
    const [, q2] = suggestedQuestions(s, 'fr');
    expect(q2.text).toContain('8 composantes');
  });
});

describe('DG-114 suggested questions — Q3 priority order', () => {
  it('event ≤ 4h → asks about event impact', () => {
    const s = makeSignal({
      event_readout: {
        news_blackout_active: false,
        next_event_label: 'CPI US',
        next_event_in_minutes: 60,
        sentiment_score: 0,
        sentiment_confidence: 0,
        session: 'london',
      },
      conviction_0_100: 85, // would otherwise trigger CI rule
    });
    const [, , q3] = suggestedQuestions(s, 'fr');
    expect(q3.source).toBe('q3_event');
    expect(q3.text).toContain('CPI US');
  });

  it('no event but conviction ≥ 70 → asks about CI', () => {
    const s = makeSignal({ conviction_0_100: 72 });
    const [, , q3] = suggestedQuestions(s, 'fr');
    expect(q3.source).toBe('q3_ci');
    expect(q3.text).toContain('[54-82]');
  });

  it('low conviction + no event → asks about regime gate', () => {
    const s = makeSignal({ conviction_0_100: 40 });
    const [, , q3] = suggestedQuestions(s, 'fr');
    expect(q3.source).toBe('q3_regime');
    expect(q3.text.toLowerCase()).toContain('régime');
    expect(q3.text).toContain('TRADE');
  });

  it('event > 4h → falls through to CI/regime', () => {
    const s = makeSignal({
      event_readout: {
        news_blackout_active: false,
        next_event_label: 'CPI US',
        next_event_in_minutes: 300, // 5h away — outside the 4h window
        sentiment_score: 0,
        sentiment_confidence: 0,
        session: 'london',
      },
      conviction_0_100: 75,
    });
    const [, , q3] = suggestedQuestions(s, 'fr');
    expect(q3.source).toBe('q3_ci');
  });
});

describe('DG-114 invariants', () => {
  it('always returns exactly 3 questions', () => {
    const s = makeSignal();
    const out = suggestedQuestions(s, 'fr');
    expect(out).toHaveLength(3);
  });

  it('each question has a non-empty text and a known source', () => {
    const s = makeSignal();
    const out = suggestedQuestions(s, 'en');
    for (const q of out) {
      expect(q.text.length).toBeGreaterThan(0);
      expect(['q1_conviction', 'q2_top_component', 'q3_event', 'q3_ci', 'q3_regime']).toContain(q.source);
    }
  });
});
