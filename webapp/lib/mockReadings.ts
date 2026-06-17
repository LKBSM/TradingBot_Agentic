/**
 * ╔══════════════════════════════════════════════════════════════════════════╗
 * ║  ⚠️  TEMPORAIRE — DONNÉES MOCK  ⚠️                                          ║
 * ║                                                                            ║
 * ║  Ce module sert UNIQUEMENT à montrer la vue /app en "produit fini" avant   ║
 * ║  que le backend (FastAPI / moteur SMC) ne serve des lectures réelles.      ║
 * ║                                                                            ║
 * ║  RIEN ici n'appelle de service externe. Aucune "API TradingView" de        ║
 * ║  données n'est utilisée (interdit). Les bougies sont générées localement   ║
 * ║  de façon déterministe (PRNG seedé) — ce ne sont PAS des cours réels.      ║
 * ║                                                                            ║
 * ║  POUR BRANCHER LE BACKEND PLUS TARD :                                       ║
 * ║    1. Passer READING_DATA_SOURCE à 'live'  → la vue /app repasse sur        ║
 * ║       fetchMarketReading() (GET /api/market-reading) au lieu des mocks.     ║
 * ║    2. Brancher un vrai flux de bougies à la place de getMockCandles()      ║
 * ║       (même forme : Candle[] trié par temps croissant, time = UNIX s).     ║
 * ║    3. Supprimer ce fichier une fois les deux flux réels en place.          ║
 * ╚══════════════════════════════════════════════════════════════════════════╝
 */

import {
  FIXTURE_EUR_H1,
  FIXTURE_QUIET_XAU_H4,
  FIXTURE_XAU_M15,
} from '@/lib/market-reading/fixtures';
import { comboKey, type Combo } from '@/lib/market-reading/store';
import type { Candle, MarketReading } from '@/types/market-reading';

// Re-exported for existing importers (ReadingChart, tests). The canonical
// definition now lives in the contract types alongside the live /api/candles
// shape — mock and live candles share one type.
export type { Candle };

/**
 * Source de données de la vue /app.
 *   · 'mock' → lectures + bougies issues de ce module (démo "produit fini").
 *   · 'live' → lectures via fetchMarketReading() (backend réel).
 * TEMPORAIRE : 'mock' tant que le groupe backend ne sert pas de données réelles.
 */
export const READING_DATA_SOURCE: 'mock' | 'live' = 'live';

// ─── Lectures mock (calquées sur le contrat MarketReading v2.0.0) ─────────────

/**
 * Lecture XAU/USD H1 — tendance haussière mûre, plusieurs zones actives.
 * (Les 3 autres réutilisent les fixtures existantes, déjà valides au schéma.)
 */
const MOCK_XAU_H1: MarketReading = {
  schema_version: '2.0.0',
  header: {
    instrument: 'XAUUSD',
    timeframe: 'H1',
    candle_close_ts: '2026-05-26T11:00:00+00:00',
    close_price: 2396.8,
  },
  structure: {
    bos: {
      direction: 'bullish',
      level: 2390.0,
      broken_at: '2026-05-26T08:00:00+00:00',
      validation_status: 'confirmed',
    },
    choch: {
      direction: 'bullish',
      level: 2372.5,
      broken_at: '2026-05-25T18:00:00+00:00',
      validation_status: 'confirmed',
    },
    order_blocks: [
      {
        id: 'ob-xau-h1-1',
        direction: 'bullish',
        level_high: 2382.0,
        level_low: 2377.0,
        importance: 'high',
        status: 'active',
        created_at: '2026-05-26T05:00:00+00:00',
        tested: false,
        user_flagged: false,
      },
    ],
    fair_value_gaps: [
      {
        id: 'fvg-xau-h1-1',
        direction: 'bullish',
        level_high: 2388.5,
        level_low: 2384.0,
        status: 'partially_filled',
        created_at: '2026-05-26T07:00:00+00:00',
        tested: true,
        user_flagged: false,
      },
    ],
    retest_in_progress: null,
  },
  regime: {
    trend: 'bullish',
    volatility_observed: 'normal',
    market_phase: 'expansion',
    mtf_confluence: { m15: 'bullish', h1: 'bullish', h4: 'bullish' },
  },
  events: {
    news_upcoming: [
      {
        event: 'US Core PCE Price Index',
        scheduled_at: '2026-05-26T12:30:00+00:00',
        time_to_event_min: 90,
        impact: 'high',
        currency: 'USD',
        potential_effect_description:
          'Indicateur d’inflation suivi par la Fed — susceptible d’élargir la volatilité sur l’or.',
      },
    ],
    news_just_published: [],
    technical_triggers_recent: [
      {
        type: 'bos_h1_bullish',
        occurred_at: '2026-05-26T08:00:00+00:00',
        minutes_ago: 180,
      },
    ],
  },
  conditions: {
    tags: ['trend', 'bos_confirmed', 'expansion'],
    description:
      'Tendance haussière en expansion sur H1, alignement M15/H1/H4. Order Block haussier non testé sous le prix, FVG partiellement comblé. Publication PCE à surveiller.',
    description_source: 'haiku_generated',
  },
};

/** Lecture EUR/USD M15 — range serré, changement de caractère en attente. */
const MOCK_EUR_M15: MarketReading = {
  schema_version: '2.0.0',
  header: {
    instrument: 'EURUSD',
    timeframe: 'M15',
    candle_close_ts: '2026-05-26T11:45:00+00:00',
    close_price: 1.08367,
  },
  structure: {
    bos: {
      direction: 'bearish',
      level: 1.0832,
      broken_at: '2026-05-26T11:00:00+00:00',
      validation_status: 'pending',
    },
    choch: null,
    order_blocks: [
      {
        id: 'ob-eur-m15-1',
        direction: 'bearish',
        level_high: 1.0848,
        level_low: 1.0844,
        importance: 'medium',
        status: 'active',
        created_at: '2026-05-26T10:15:00+00:00',
        tested: false,
        user_flagged: false,
      },
    ],
    fair_value_gaps: [
      {
        id: 'fvg-eur-m15-1',
        direction: 'bearish',
        level_high: 1.0841,
        level_low: 1.0838,
        status: 'active',
        created_at: '2026-05-26T11:15:00+00:00',
        tested: false,
        user_flagged: false,
      },
    ],
    retest_in_progress: {
      level: 1.0832,
      type: 'bos_retest',
      started_at: '2026-05-26T11:30:00+00:00',
    },
  },
  regime: {
    trend: 'bearish',
    volatility_observed: 'normal',
    market_phase: 'distribution',
    mtf_confluence: { m15: 'bearish', h1: 'neutral', h4: 'bearish' },
  },
  events: {
    news_upcoming: [],
    news_just_published: [],
    technical_triggers_recent: [
      {
        type: 'bos_m15_bearish',
        occurred_at: '2026-05-26T11:00:00+00:00',
        minutes_ago: 45,
      },
    ],
  },
  conditions: {
    tags: ['distribution', 'bos_pending', 'retest_active'],
    description:
      'Cassure baissière récente en attente de confirmation, retest du niveau cassé en cours. Phase de distribution, biais H4 baissier.',
    description_source: 'haiku_generated',
  },
};

/** Lecture EUR/USD H4 — calme, peu de structure (exerce les sections "vides"). */
const MOCK_EUR_H4: MarketReading = {
  schema_version: '2.0.0',
  header: {
    instrument: 'EURUSD',
    timeframe: 'H4',
    candle_close_ts: '2026-05-26T08:00:00+00:00',
    close_price: 1.0855,
  },
  structure: {
    bos: null,
    choch: {
      direction: 'bearish',
      level: 1.0872,
      broken_at: '2026-05-23T16:00:00+00:00',
      validation_status: 'confirmed',
    },
    order_blocks: [
      {
        id: 'ob-eur-h4-1',
        direction: 'bearish',
        level_high: 1.0885,
        level_low: 1.0872,
        importance: 'high',
        status: 'active',
        created_at: '2026-05-23T12:00:00+00:00',
        tested: false,
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
    mtf_confluence: { h1: 'neutral', h4: 'bearish' },
  },
  events: {
    news_upcoming: [],
    news_just_published: [],
    technical_triggers_recent: [],
  },
  conditions: {
    tags: ['ranging', 'low_vol'],
    description:
      'Marché en range sur H4, volatilité basse. Order Block baissier en surplomb non encore testé, biais structurel baissier de fond.',
    description_source: 'template_fallback',
  },
};

/**
 * Toutes les lectures mock, indexées par clé de combo. Le périmètre V1 (6 combos)
 * est entièrement couvert pour que la vue /app paraisse "finie".
 */
const MOCK_READINGS: Readonly<Record<string, MarketReading>> = {
  [comboKey({ instrument: 'XAUUSD', timeframe: 'M15' })]: FIXTURE_XAU_M15,
  [comboKey({ instrument: 'XAUUSD', timeframe: 'H1' })]: MOCK_XAU_H1,
  [comboKey({ instrument: 'XAUUSD', timeframe: 'H4' })]: FIXTURE_QUIET_XAU_H4,
  [comboKey({ instrument: 'EURUSD', timeframe: 'M15' })]: MOCK_EUR_M15,
  [comboKey({ instrument: 'EURUSD', timeframe: 'H1' })]: FIXTURE_EUR_H1,
  [comboKey({ instrument: 'EURUSD', timeframe: 'H4' })]: MOCK_EUR_H4,
};

/** Lecture mock pour un combo, ou null si non couvert (→ état "indisponible"). */
export function getMockReading(
  instrument: string,
  timeframe: string,
): MarketReading | null {
  return MOCK_READINGS[comboKey({ instrument, timeframe })] ?? null;
}

// ─── Bougies mock (pour le graphique) ────────────────────────────────────────
// La forme `Candle` est définie dans les types du contrat (cf. import en tête) :
// `time` = timestamp UNIX en SECONDES (UTCTimestamp).

const INTERVAL_SECONDS: Record<string, number> = {
  M15: 15 * 60,
  H1: 60 * 60,
  H4: 4 * 60 * 60,
};

/** PRNG déterministe (mulberry32) — bougies stables entre les rendus / tests. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Seed stable dérivée d'une chaîne (clé de combo). */
function seedFrom(key: string): number {
  let h = 2166136261;
  for (let i = 0; i < key.length; i += 1) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * Génère une marche aléatoire OHLC déterministe se terminant exactement sur
 * `lastClose`, dont l'enveloppe [low, high] englobe tous les `levels` fournis
 * (niveaux BOS/CHOCH/OB/FVG) pour que les overlays tombent bien dans le cadre.
 */
function generateCandles(
  key: string,
  count: number,
  intervalSec: number,
  lastCloseTs: number,
  lastClose: number,
  levels: number[],
): Candle[] {
  const rng = mulberry32(seedFrom(key));

  const lo = Math.min(lastClose, ...levels);
  const hi = Math.max(lastClose, ...levels);
  const center = (lo + hi) / 2;
  const span = Math.max(hi - lo, lastClose * 0.006);
  const amplitude = span * 0.85;
  const step = amplitude / 6;

  // Marche aléatoire autour du centre, rappel doux vers le centre.
  const closes: number[] = [];
  let price = center - amplitude * 0.4;
  for (let i = 0; i < count; i += 1) {
    const pull = (center - price) * 0.06;
    price += (rng() - 0.5) * 2 * step + pull;
    closes.push(price);
  }

  // Recale pour que la dernière clôture vaille exactement lastClose.
  const shift = lastClose - closes[count - 1]!;
  for (let i = 0; i < count; i += 1) closes[i]! += shift;

  const candles: Candle[] = [];
  for (let i = 0; i < count; i += 1) {
    const close = closes[i]!;
    const open = i === 0 ? close - (rng() - 0.5) * step : closes[i - 1]!;
    const wickUp = rng() * step * 0.7;
    const wickDn = rng() * step * 0.7;
    const high = Math.max(open, close) + wickUp;
    const low = Math.min(open, close) - wickDn;
    const time = lastCloseTs - (count - 1 - i) * intervalSec;
    candles.push({ time, open, high, low, close });
  }

  // Garantit que l'enveloppe englobe chaque niveau (overlays toujours visibles).
  if (candles.length > 0) {
    const margin = span * 0.04;
    let maxHighIdx = 0;
    let minLowIdx = 0;
    for (let i = 1; i < candles.length; i += 1) {
      if (candles[i]!.high > candles[maxHighIdx]!.high) maxHighIdx = i;
      if (candles[i]!.low < candles[minLowIdx]!.low) minLowIdx = i;
    }
    candles[maxHighIdx]!.high = Math.max(candles[maxHighIdx]!.high, hi + margin);
    candles[minLowIdx]!.low = Math.min(candles[minLowIdx]!.low, lo - margin);
  }

  return candles;
}

/** Combos dont le flux de bougies est volontairement indisponible (démo état d'erreur). */
const CANDLE_FEED_UNAVAILABLE: ReadonlySet<string> = new Set<string>([
  // XAU/USD H4 : la lecture textuelle reste disponible, mais le flux de
  // bougies n'est pas connecté pour cette combinaison → démontre le placeholder
  // "Graphique indisponible" en conditions réelles (dégradation gracieuse).
  comboKey({ instrument: 'XAUUSD', timeframe: 'H4' }),
]);

/**
 * Bougies mock pour un combo, ou `null` si le flux n'est pas disponible
 * (combo hors catalogue OU feed volontairement indisponible). Les niveaux de la
 * lecture servent à cadrer l'enveloppe de prix.
 */
export function getMockCandles(
  instrument: string,
  timeframe: string,
): Candle[] | null {
  const key = comboKey({ instrument, timeframe });
  if (CANDLE_FEED_UNAVAILABLE.has(key)) return null;

  const reading = MOCK_READINGS[key];
  const intervalSec = INTERVAL_SECONDS[timeframe];
  if (!reading || !intervalSec) return null;

  const levels = collectLevels(reading);
  const lastCloseTs = Math.floor(
    new Date(reading.header.candle_close_ts).getTime() / 1000,
  );

  return generateCandles(
    key,
    96,
    intervalSec,
    lastCloseTs,
    reading.header.close_price,
    levels,
  );
}

/** Rassemble tous les niveaux de prix d'une lecture (pour cadrer le graphique). */
function collectLevels(reading: MarketReading): number[] {
  const s = reading.structure;
  const levels: number[] = [];
  if (s.bos) levels.push(s.bos.level);
  if (s.choch) levels.push(s.choch.level);
  if (s.retest_in_progress) levels.push(s.retest_in_progress.level);
  for (const ob of s.order_blocks) levels.push(ob.level_high, ob.level_low);
  for (const fvg of s.fair_value_gaps) levels.push(fvg.level_high, fvg.level_low);
  return levels.length > 0 ? levels : [reading.header.close_price];
}

/** Combo par défaut au chargement de /app (lecture XAU M15). */
export const DEFAULT_COMBO: Combo = { instrument: 'XAUUSD', timeframe: 'M15' };
