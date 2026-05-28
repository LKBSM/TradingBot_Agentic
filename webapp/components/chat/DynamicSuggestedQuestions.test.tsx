import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { DynamicSuggestedQuestions } from './DynamicSuggestedQuestions';
import type { InsightSignalV2 } from '@/types/insight';

function makeSignal(overrides: Partial<InsightSignalV2> = {}): InsightSignalV2 {
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
      bos_level: 2350, bos_event_age_bars: 3, choch_present: true,
      fvg_zone: null, fvg_size_atr: null, ob_zone: null, ob_strength: null,
      retest_state: 'armed', structural_invalidation: 2340,
      liquidity_zone_upper: null, liquidity_zone_lower: null,
    },
    regime_readout: {
      hmm_label: 'trend_bullish', hmm_posterior: 0.7, bocpd_changepoint_prob: 0.05,
      expected_run_length: 24, jump_ratio: 0.1, regime_gate_decision: 'TRADE',
    },
    volatility_readout: {
      regime: 'normal', forecast_atr_pips: 12, naive_atr_pips: 14,
      forecast_vs_naive_pct: -14, confidence_interval_pips: [10, 14], is_fallback: false,
    },
    event_readout: {
      news_blackout_active: false, next_event_label: null, next_event_in_minutes: null,
      sentiment_score: 0, sentiment_confidence: 0, session: 'london',
    },
    breakdown_components: [
      { name: 'bos', contribution: 9, weight_max: 15, reasoning: 'BOS' },
      { name: 'order_block', contribution: 12, weight_max: 15, reasoning: 'OB' },
    ],
    historical_stats: null,
    narrative_short: 'test',
    narrative_long: null,
  } as unknown as InsightSignalV2;
  return { ...base, ...overrides };
}

describe('DynamicSuggestedQuestions — DG-114', () => {
  it('renders exactly 3 chips derived from the signal', () => {
    const onAsk = vi.fn();
    render(
      <DynamicSuggestedQuestions
        signal={makeSignal()}
        onAsk={onAsk}
      />,
    );
    const chips = screen.getAllByRole('button');
    expect(chips).toHaveLength(3);
  });

  it('clicking a chip calls onAsk with the question text', () => {
    const onAsk = vi.fn();
    render(
      <DynamicSuggestedQuestions
        signal={makeSignal()}
        onAsk={onAsk}
      />,
    );
    const chips = screen.getAllByRole('button');
    const first = chips[0];
    expect(first).toBeDefined();
    fireEvent.click(first!);
    expect(onAsk).toHaveBeenCalledTimes(1);
    const arg = onAsk.mock.calls[0]?.[0];
    expect(typeof arg).toBe('string');
    expect(arg.length).toBeGreaterThan(0);
  });

  it('disabled prop blocks clicks and dims chips', () => {
    const onAsk = vi.fn();
    render(
      <DynamicSuggestedQuestions
        signal={makeSignal()}
        onAsk={onAsk}
        disabled
      />,
    );
    const chips = screen.getAllByRole('button');
    chips.forEach((c) => expect(c).toBeDisabled());
    fireEvent.click(chips[0]!);
    expect(onAsk).not.toHaveBeenCalled();
  });

  it('Q1 chip carries data-source="q1_conviction"', () => {
    render(
      <DynamicSuggestedQuestions
        signal={makeSignal()}
        onAsk={() => {}}
      />,
    );
    const q1 = screen
      .getAllByRole('button')
      .find((b) => b.getAttribute('data-source') === 'q1_conviction');
    expect(q1).toBeDefined();
  });
});
