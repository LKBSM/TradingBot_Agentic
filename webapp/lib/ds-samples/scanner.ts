/**
 * DS-1 — Frozen scanner sample data for the design gallery.
 *
 * The ConditionsScanResponse shape mirrors the backend contract
 * (@/lib/conditions/types). Every combo's CONTEXT (trend, phase, volatility,
 * active OB/FVG counts, close price, candle_close_ts, MTF units) is copied from
 * the REAL readings in market_readings.db — the same rows that feed readings.ts:
 *   · XAUUSD H4  2026-07-31T20:00Z — bearish, phase trend, vol normal, 5 OB / 8 FVG
 *   · EURUSD M15 2026-07-31T21:00Z — bullish, phase trend, vol low,    4 OB / 10 FVG
 *   · XAUUSD D1  2026-07-31T00:00Z — bearish, phase trend, vol normal, 6 OB / 5 FVG
 *
 * The condition outcome labels/details are the descriptive, present-tense strings
 * the backend attaches — kept factual, no forbidden vocabulary. No network.
 */
import type { ConditionsConfig, ConditionsScanResponse } from '@/lib/conditions/types';

/** The saved reading driving the scan: three present-tense structural conditions. */
export const SAMPLE_SCAN_CONFIG: ConditionsConfig = {
  logic: 'AND',
  conditions: [
    { type: 'trend_is', trend: 'bearish' },
    { type: 'higher_tf_agrees', relation: 'same' },
    { type: 'price_near_fvg', proximity_pct: 0.3 },
  ],
};

/**
 * A scan with a full match (XAUUSD H4), a partial (EURUSD M15), a partial with a
 * non-evaluable condition (XAUUSD D1), and an unavailable combo. Exercises every
 * ScanResults group: Correspondances / Presque / Non évaluables + Contexte.
 */
export const SAMPLE_SCAN_RESPONSE: ConditionsScanResponse = {
  as_of: '2026-07-31T21:00:00Z',
  logic: 'AND',
  scanned: 4,
  matches: [
    {
      instrument: 'XAUUSD',
      timeframe: 'H4',
      candle_close_ts: '2026-07-31T20:00:00Z',
      close_price: 4043.24,
      matched: true,
      met_count: 3,
      total: 3,
      non_evaluable_count: 0,
      freshness: 'fresh',
      conditions_met: [
        { type: 'trend_is', label: 'Tendance baissière', met: true, detail: 'Structure orientée à la baisse depuis le CHOCH du 24 juil.' },
        { type: 'higher_tf_agrees', label: 'Le timeframe supérieur concorde', met: true, detail: 'D1 orienté à la baisse, dans le même sens que H4.' },
        { type: 'price_near_fvg', label: 'Prix proche d’une FVG', met: true, detail: 'Une FVG baissière active se situe à 0,24 % au-dessus du prix.' },
      ],
      conditions_unmet: [],
      context_against: [
        { label: 'Zone opposée active', detail: 'Un Order Block haussier reste actif sous le prix — orientation inverse de la tendance.' },
      ],
      context: {
        trend: 'bearish',
        market_phase: 'trend',
        volatility_observed: 'normal',
        mtf_confluence: { d1: 'bearish' },
        mtf_trends: { d1: 'bearish' },
        bos: null,
        choch: { direction: 'bearish', level: 4044.44, validation_status: 'confirmed' },
        active_order_blocks: 5,
        active_fair_value_gaps: 8,
        structural_range: { low: 3942.48, high: 4382.48 },
        news_upcoming: [],
      },
      bars_behind: 0,
    },
    {
      instrument: 'EURUSD',
      timeframe: 'M15',
      candle_close_ts: '2026-07-31T21:00:00Z',
      close_price: 1.15312,
      matched: false,
      met_count: 2,
      total: 3,
      non_evaluable_count: 0,
      freshness: 'fresh',
      conditions_met: [
        { type: 'higher_tf_agrees', label: 'Le timeframe supérieur concorde', met: true, detail: 'H1 et H4 orientés à la hausse, dans le même sens que M15.' },
        { type: 'price_near_fvg', label: 'Prix proche d’une FVG', met: true, detail: 'Une FVG haussière active se situe à 0,10 % sous le prix.' },
      ],
      conditions_unmet: [
        { type: 'trend_is', label: 'Tendance baissière', met: false, detail: 'La structure est orientée à la hausse sur M15.' },
      ],
      context: {
        trend: 'bullish',
        market_phase: 'trend',
        volatility_observed: 'low',
        mtf_confluence: { h1: 'bullish', h4: 'bullish' },
        mtf_trends: { h1: 'bullish', h4: 'bullish' },
        bos: { direction: 'bullish', level: 1.15193, validation_status: 'confirmed' },
        choch: null,
        active_order_blocks: 4,
        active_fair_value_gaps: 10,
        structural_range: { low: 1.14343, high: 1.15376 },
        news_upcoming: [],
      },
      bars_behind: 0,
    },
    {
      instrument: 'XAUUSD',
      timeframe: 'D1',
      candle_close_ts: '2026-07-31T00:00:00Z',
      close_price: 4111.96,
      matched: false,
      met_count: 1,
      total: 2,
      non_evaluable_count: 1,
      freshness: 'fresh',
      conditions_met: [
        { type: 'trend_is', label: 'Tendance baissière', met: true, detail: 'Structure orientée à la baisse depuis le CHOCH du 10 juin.' },
      ],
      conditions_unmet: [
        { type: 'price_near_fvg', label: 'Prix proche d’une FVG', met: false, detail: 'Aucune FVG active à moins de 0,3 % du prix.' },
      ],
      conditions_non_evaluable: [
        { type: 'higher_tf_agrees', label: 'Le timeframe supérieur concorde', met: false, available: false, detail: 'Aucun timeframe supérieur disponible au-dessus de D1.' },
      ],
      context: {
        trend: 'bearish',
        market_phase: 'trend',
        volatility_observed: 'normal',
        mtf_confluence: {},
        mtf_trends: {},
        bos: null,
        choch: { direction: 'bearish', level: 4098.53, validation_status: 'confirmed' },
        active_order_blocks: 6,
        active_fair_value_gaps: 5,
        structural_range: { low: 3886.62, high: 5598.03 },
        news_upcoming: [],
      },
      bars_behind: 0,
    },
  ],
  unavailable: [
    { instrument: 'BTCUSD', timeframe: 'H1', reason: 'Aucune lecture récente pour cette combinaison.' },
  ],
};

/**
 * Same conditions, but no combo satisfies them in full — drives the explicit
 * "ce n'est pas une erreur" state with per-condition ISOLATED counts (VZ rule:
 * no full match → say so, never loosen a condition silently).
 */
export const SAMPLE_SCAN_NO_MATCH: ConditionsScanResponse = {
  as_of: '2026-07-31T21:00:00Z',
  logic: 'AND',
  scanned: 2,
  matches: [
    { ...SAMPLE_SCAN_RESPONSE.matches[1]! },
    { ...SAMPLE_SCAN_RESPONSE.matches[2]! },
  ],
  unavailable: [
    { instrument: 'BTCUSD', timeframe: 'H1', reason: 'Aucune lecture récente pour cette combinaison.' },
  ],
};
