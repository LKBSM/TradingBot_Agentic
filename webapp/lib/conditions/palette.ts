/**
 * Conditions Scanner — the palette (single source of truth for the builder).
 *
 * Every entry is a STRUCTURAL FACT AT THE PRESENT, derivable from the existing
 * MarketReading (regime + structure). There is intentionally no predictive /
 * outcome condition ("will bounce", "will break"): such a thing is not
 * representable here, and a test asserts the palette stays present-tense.
 *
 * Mirrors the backend palette (src/intelligence/conditions_scanner.py:PALETTE).
 */

import type {
  ConditionType,
  LiquiditySideFilter,
  PaletteEntry,
  PhaseChoice,
  TrendChoice,
  VolatilityChoice,
} from './types';

export const CONDITION_PALETTE: readonly PaletteEntry[] = [
  {
    type: 'mtf_aligned',
    label: '3 TF alignés',
    description:
      'Les 3 timeframes (H4, H1, M15) pointent dans la même direction en ce moment.',
    controls: ['direction'],
    tense: 'present',
  },
  {
    type: 'trend_is',
    label: 'Tendance actuelle',
    description:
      'La tendance observée sur ce timeframe est, en ce moment, celle choisie.',
    controls: ['trend'],
    tense: 'present',
  },
  {
    type: 'market_phase_is',
    label: 'Phase de marché',
    description:
      'La phase de marché observée correspond, en ce moment, à celle choisie.',
    controls: ['phase'],
    tense: 'present',
  },
  {
    type: 'volatility_is',
    label: 'Volatilité observée',
    description:
      'La volatilité observée en ce moment correspond au niveau choisi.',
    controls: ['volatility'],
    tense: 'present',
  },
  {
    type: 'price_in_ob',
    label: 'Prix dans un Order Block',
    description: 'Le prix courant se situe à l’intérieur d’un Order Block actif.',
    controls: ['direction'],
    tense: 'present',
  },
  {
    type: 'price_in_fvg',
    label: 'Prix dans un Fair Value Gap',
    description:
      'Le prix courant se situe à l’intérieur d’un Fair Value Gap non comblé.',
    controls: ['direction'],
    tense: 'present',
  },
  {
    type: 'ob_fvg_confluence',
    label: 'Confluence OB + FVG au prix courant',
    description:
      'Le prix courant se situe simultanément dans un Order Block actif et dans un Fair Value Gap non comblé.',
    controls: [],
    tense: 'present',
  },
  {
    type: 'bos_recent_confirmed',
    label: 'BOS confirmé récent',
    description:
      'Une cassure de structure (BOS) confirmée est datée des dernières bougies.',
    controls: ['direction', 'bars'],
    tense: 'present',
  },
  {
    type: 'choch_recent_confirmed',
    label: 'CHOCH confirmé récent',
    description:
      'Un changement de caractère (CHOCH) confirmé est daté des dernières bougies.',
    controls: ['direction', 'bars'],
    tense: 'present',
  },
  {
    type: 'retest_in_progress',
    label: 'Retest en cours',
    description:
      'Un retest d’un niveau (BOS, CHOCH, OB ou FVG) est en cours en ce moment.',
    controls: [],
    tense: 'present',
  },
  {
    type: 'price_near_ob',
    label: 'Prix proche d’un Order Block',
    description:
      'Le prix courant est proche d’un Order Block actif (sous la distance choisie), sans être nécessairement dedans.',
    controls: ['direction', 'proximity'],
    tense: 'present',
  },
  {
    type: 'price_near_fvg',
    label: 'Prix proche d’un Fair Value Gap',
    description:
      'Le prix courant est proche d’un Fair Value Gap non comblé (sous la distance choisie).',
    controls: ['direction', 'proximity'],
    tense: 'present',
  },
  {
    type: 'price_near_liquidity',
    label: 'Prix proche d’une liquidité (SSL/BSL)',
    description:
      'Le prix courant est proche d’une poche de liquidité intacte (BSL au-dessus / SSL en dessous).',
    controls: ['side', 'proximity'],
    tense: 'present',
  },
  {
    type: 'liquidity_swept_recent',
    label: 'Prise de liquidité récente',
    description:
      'Une poche de liquidité (SSL/BSL) a été balayée au cours des dernières bougies.',
    controls: ['side', 'bars'],
    tense: 'present',
  },
] as const;

export const CONDITION_TYPES: readonly ConditionType[] = CONDITION_PALETTE.map(
  (p) => p.type,
);

export function paletteEntry(type: ConditionType): PaletteEntry | undefined {
  return CONDITION_PALETTE.find((p) => p.type === type);
}

export const DIRECTION_LABELS: Record<string, string> = {
  any: 'Toute direction',
  bullish: 'Haussier',
  bearish: 'Baissier',
};

export const TREND_OPTIONS: Array<{ value: TrendChoice; label: string }> = [
  { value: 'bullish', label: 'Haussière' },
  { value: 'bearish', label: 'Baissière' },
  { value: 'ranging', label: 'Range' },
  { value: 'neutral', label: 'Neutre' },
];

export const PHASE_OPTIONS: Array<{ value: PhaseChoice; label: string }> = [
  { value: 'accumulation', label: 'Accumulation' },
  { value: 'distribution', label: 'Distribution' },
  { value: 'trend', label: 'Tendance' },
  { value: 'ranging', label: 'Range' },
  { value: 'expansion', label: 'Expansion' },
];

export const VOLATILITY_OPTIONS: Array<{ value: VolatilityChoice; label: string }> = [
  { value: 'low', label: 'Faible' },
  { value: 'normal', label: 'Normale' },
  { value: 'elevated', label: 'Élevée' },
];

export const LIQUIDITY_SIDE_OPTIONS: Array<{
  value: LiquiditySideFilter;
  label: string;
}> = [
  { value: 'any', label: 'Les deux (SSL + BSL)' },
  { value: 'bsl', label: 'BSL (liquidité au-dessus)' },
  { value: 'ssl', label: 'SSL (liquidité en dessous)' },
];

export const DEFAULT_BOS_MAX_BARS = 5;
/** Default proximity threshold (% of price) for the "price near …" conditions. */
export const DEFAULT_PROXIMITY_PCT = 0.3;
/** Default recency window (bars) for liquidity_swept_recent. */
export const DEFAULT_LIQ_MAX_BARS = 10;
