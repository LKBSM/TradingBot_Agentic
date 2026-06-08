import type { MarketReading } from '@/types/market-reading';

/**
 * Hand-curated MarketReading fixtures that validate against the Pydantic v2.0.0
 * schema. NOT live data — they exercise the UI components and the /app view
 * before (and alongside) the backend integration. When the real endpoint is
 * wired (Étape 2), these remain available for tests and Storybook-style usage.
 *
 * Timestamps are fixed (not `now`-relative) so snapshot/age assertions stay
 * deterministic; components compute relative ages against the live clock.
 */

export const FIXTURE_XAU_M15: MarketReading = {
  schema_version: '2.0.0',
  header: {
    instrument: 'XAUUSD',
    timeframe: 'M15',
    candle_close_ts: '2026-05-26T11:45:00+00:00',
    close_price: 2392.35,
  },
  structure: {
    bos: {
      direction: 'bullish',
      level: 2391.5,
      broken_at: '2026-05-26T11:15:00+00:00',
      validation_status: 'confirmed',
    },
    choch: {
      direction: 'bullish',
      level: 2384.2,
      broken_at: '2026-05-26T09:30:00+00:00',
      validation_status: 'confirmed',
    },
    order_blocks: [
      {
        id: 'ob-xau-1',
        direction: 'bullish',
        level_high: 2378.0,
        level_low: 2375.0,
        importance: 'high',
        status: 'active',
        created_at: '2026-05-26T08:00:00+00:00',
        tested: false,
        user_flagged: false,
      },
    ],
    fair_value_gaps: [
      {
        id: 'fvg-xau-1',
        direction: 'bullish',
        level_high: 2381.0,
        level_low: 2378.0,
        status: 'active',
        created_at: '2026-05-26T10:45:00+00:00',
        tested: false,
        user_flagged: false,
      },
    ],
    retest_in_progress: {
      level: 2391.5,
      type: 'bos_retest',
      started_at: '2026-05-26T11:30:00+00:00',
    },
  },
  regime: {
    trend: 'bullish',
    volatility_observed: 'normal',
    market_phase: 'trend',
    mtf_confluence: {
      m15: 'bullish',
      h1: 'bullish',
      h4: 'neutral',
    },
  },
  events: {
    news_upcoming: [
      {
        event: 'FOMC Minutes',
        scheduled_at: '2026-05-26T18:00:00+00:00',
        time_to_event_min: 375,
        impact: 'high',
        currency: 'USD',
        potential_effect_description:
          'Publication susceptible d’élargir la volatilité sur les paires USD et l’or.',
      },
    ],
    news_just_published: [],
    technical_triggers_recent: [
      {
        type: 'bos_m15_bullish',
        occurred_at: '2026-05-26T11:15:00+00:00',
        minutes_ago: 30,
      },
    ],
  },
  conditions: {
    tags: ['trend', 'bos_confirmed', 'retest_active'],
    description:
      'Structure haussière confirmée par une cassure récente, retest en cours sur le niveau cassé. Volatilité normale, alignement M15/H1.',
    description_source: 'haiku_generated',
  },
};

export const FIXTURE_EUR_H1: MarketReading = {
  schema_version: '2.0.0',
  header: {
    instrument: 'EURUSD',
    timeframe: 'H1',
    candle_close_ts: '2026-05-26T11:00:00+00:00',
    close_price: 1.08423,
  },
  structure: {
    bos: null,
    choch: {
      direction: 'bearish',
      level: 1.0865,
      broken_at: '2026-05-26T08:00:00+00:00',
      validation_status: 'pending',
    },
    order_blocks: [
      {
        id: 'ob-eur-1',
        direction: 'bearish',
        level_high: 1.0871,
        level_low: 1.0858,
        importance: 'medium',
        status: 'mitigated',
        created_at: '2026-05-26T05:00:00+00:00',
        tested: true,
        user_flagged: false,
      },
    ],
    fair_value_gaps: [],
    retest_in_progress: null,
  },
  regime: {
    trend: 'ranging',
    volatility_observed: 'low',
    market_phase: 'ranging',
    mtf_confluence: {
      m15: 'neutral',
      h1: 'ranging',
      h4: 'bearish',
    },
  },
  events: {
    news_upcoming: [],
    news_just_published: [
      {
        event: 'German Ifo Business Climate',
        published_at: '2026-05-26T08:00:00+00:00',
        actual: 88.6,
        forecast: 89.4,
        previous: 89.2,
        surprise_direction: 'miss',
        currency: 'EUR',
        impact: 'medium',
        potential_effect_description:
          'Donnée légèrement sous le consensus, pression descendante modérée sur l’euro.',
      },
    ],
    technical_triggers_recent: [],
  },
  conditions: {
    tags: ['ranging', 'choch_pending', 'low_vol'],
    description:
      'Marché en range sur H1, volatilité basse. Changement de caractère baissier en attente de confirmation, biais H4 baissier.',
    description_source: 'template_fallback',
  },
};

/** Minimal reading with empty structure/events — exercises empty-state rendering. */
export const FIXTURE_QUIET_XAU_H4: MarketReading = {
  schema_version: '2.0.0',
  header: {
    instrument: 'XAUUSD',
    timeframe: 'H4',
    candle_close_ts: '2026-05-26T08:00:00+00:00',
    close_price: 2388.1,
  },
  structure: {
    bos: null,
    choch: null,
    order_blocks: [],
    fair_value_gaps: [],
    retest_in_progress: null,
  },
  regime: {
    trend: 'neutral',
    volatility_observed: 'low',
    market_phase: 'accumulation',
    mtf_confluence: { h4: 'neutral' },
  },
  events: {
    news_upcoming: [],
    news_just_published: [],
    technical_triggers_recent: [],
  },
  conditions: {
    tags: ['quiet'],
    description: 'Pas d’événement structurel notable sur la dernière bougie H4.',
    description_source: 'template_fallback',
  },
};

export const SAMPLE_READINGS: readonly MarketReading[] = [
  FIXTURE_XAU_M15,
  FIXTURE_EUR_H1,
  FIXTURE_QUIET_XAU_H4,
];
