/**
 * LP-1 — FROZEN illustration data for the home-page interactive demos.
 *
 * Everything here is a fixed, deterministic sample. The demos NEVER call the
 * network and NEVER touch a session — the whole landing must render on a plain
 * static host. A visible mention ("données d'illustration") accompanies every
 * demo so a made-up sample is never mistaken for a real market reading.
 *
 * No Math.random / Date.now: the series is hard-coded so server and client
 * render identical HTML (no hydration mismatch).
 */

/** Gold-like close series (illustration). Rises through a swept low, a CHOCH,
 * a BOS, into an untested Order Block near the current price. Matches the
 * narrated facts used by the structure demo. */
export const DEMO_CLOSES: readonly number[] = [
  4014.2, 4013.1, 4012.4, 4013.6, 4011.9, 4010.2, 4009.1, 4008.3, 4009.7, 4011.4,
  4010.8, 4012.9, 4014.6, 4013.8, 4015.2, 4016.9, 4015.7, 4017.8, 4019.3, 4018.6,
  4020.1, 4021.5, 4020.8, 4022.4, 4021.9, 4023.6, 4022.8, 4024.1, 4025.1, 4024.4,
  4025.9, 4026.3, 4025.6, 4026.8, 4027.4, 4026.6, 4027.9, 4026.9, 4027.6, 4026.2,
  4027.3, 4026.5, 4027.1, 4026.77,
];

/** The reference price band lines drawn on the structure demo. */
export const DEMO_LEVELS = {
  currentPrice: 4026.77,
  obLow: 4026.8,
  obHigh: 4028.9,
  fvgLow: 4020.1,
  fvgHigh: 4021.85,
  liqIntact: 4034.2, // buy-side, above — never reached (intact)
  liqSwept: 4011.4, // sell-side, below — swept
  chochLevel: 4021.45,
  bosLevel: 4025.1,
} as const;

export type LayerKey = 'ob' | 'fvg' | 'liq' | 'str';

/** Scanner demo — five real conditions (exact product labels come from i18n).
 * Each market row carries a 0/1 vector aligned to the five condition keys. */
export const DEMO_CONDITION_KEYS = [
  'trend_bullish', // La tendance structurelle est haussière
  'higher_tf_agrees', // L'unité supérieure va dans le même sens
  'price_in_ob', // Le prix est dans un Order Block
  'zone_untested', // La zone n'a jamais été testée
  'liquidity_swept', // Une poche a été prise récemment
] as const;

export type DemoConditionKey = (typeof DEMO_CONDITION_KEYS)[number];

export interface DemoMarketRow {
  /** i18n key under home.demo.scanner.markets.* */
  key: string;
  /** short symbol glyph */
  sym: string;
  price: string;
  /** 0/1 per DEMO_CONDITION_KEYS */
  c: readonly [number, number, number, number, number];
}

/** Two markets × timeframes — the REAL perimeter (Or, Euro). Illustration rows
 * spread across the timeframes the product actually scans. */
export const DEMO_MARKETS: readonly DemoMarketRow[] = [
  { key: 'gold_m15', sym: 'Au', price: '4 027,36', c: [1, 1, 1, 1, 0] },
  { key: 'gold_h1', sym: 'Au', price: '4 027,36', c: [1, 1, 1, 1, 0] },
  { key: 'euro_m15', sym: '€', price: '1,0842', c: [1, 1, 1, 0, 1] },
  { key: 'euro_h1', sym: '€', price: '1,0839', c: [1, 1, 0, 0, 1] },
  { key: 'gold_h4', sym: 'Au', price: '4 026,90', c: [1, 0, 1, 1, 0] },
  { key: 'euro_h4', sym: '€', price: '1,0851', c: [1, 1, 0, 1, 0] },
];

export type ZoneState = 'untested' | 'tested' | 'filled';

export interface DemoZone {
  /** i18n key under home.demo.zones.z.* */
  key: string;
  state: ZoneState;
  kind: 'ob' | 'fvg';
  dir: 'up' | 'down';
  band: string;
  /** fill % for partially filled FVG (0..100) */
  fill?: number;
}

export const DEMO_ZONES: readonly DemoZone[] = [
  { key: 'untested', state: 'untested', kind: 'ob', dir: 'up', band: '4 026,80 – 4 028,90' },
  { key: 'tested', state: 'tested', kind: 'fvg', dir: 'down', band: '4 020,10 – 4 021,85', fill: 60 },
  { key: 'filled', state: 'filled', kind: 'ob', dir: 'down', band: '4 014,20 – 4 016,05' },
];

export type MiaKind = 'answer' | 'refusal' | 'action';

export interface DemoMiaExchange {
  /** i18n key under home.demo.mia.q.* (question + answer + optional note) */
  key: string;
  kind: MiaKind;
  /** for action exchanges: the layer view to apply to the structure demo */
  action?: { only: LayerKey[] };
}

/** MIA demo — one concept answer, one grounded description, one ACTION that
 * really changes the structure demo's layers, one refusal. */
export const DEMO_MIA: readonly DemoMiaExchange[] = [
  { key: 'concept', kind: 'answer' },
  { key: 'grounded', kind: 'answer' },
  { key: 'action_ob', kind: 'action', action: { only: ['ob'] } },
  { key: 'refusal', kind: 'refusal' },
];

/** Régime tile "Ouvre le calcul" demo — a verdict that flips to the raw numbers
 * that produced it, plus what it does NOT say. Values are illustration. */
export const DEMO_REGIME = {
  recentAtr: '2,10',
  baselineAtr: '1,95',
  ratio: '1,08',
  lowThreshold: '0,80',
  highThreshold: '1,25',
} as const;
