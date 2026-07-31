/**
 * Conditions Scanner — the palette (SC-1), grouped in four families.
 *
 * Every entry is a STRUCTURAL FACT AT THE PRESENT (or a past event counted in
 * BARS). There is intentionally no predictive / outcome condition: such a thing
 * is not representable here, and a test asserts the palette stays present-tense.
 *
 * This mirrors the backend palette (src/intelligence/conditions_scanner.py:PALETTE)
 * — same types, families and controls — and is the fallback the builder renders.
 * The authoritative copy is fetched from GET /api/conditions-scan/palette so the
 * interface derives from the schema; a vitest asserts the two never drift.
 */

import type {
  ConditionType,
  ControlName,
  Family,
  PaletteEntry,
} from './types';

/** Fixed family order (never sorted, never by match count). */
export const FAMILIES: readonly Family[] = ['structure', 'zones', 'liquidity', 'context'] as const;

/** Families open by default in the builder. */
export const DEFAULT_OPEN_FAMILIES: readonly Family[] = ['structure', 'zones'] as const;

// Segmented value sets (mirror the backend vocabulary).
const DIRECTION = ['any', 'bullish', 'bearish'] as const;
const TREND = ['bullish', 'bearish', 'indeterminate'] as const;
const PHASE = ['accumulation', 'trend', 'ranging', 'expansion'] as const;
const VOLATILITY = ['low', 'normal', 'elevated'] as const;
const SIDE = ['any', 'bsl', 'ssl'] as const;
const ZONE_KIND = ['any', 'ob', 'fvg'] as const;
const EVENT = ['bos_up', 'bos_down', 'choch_up', 'choch_down'] as const;
const AGE_BUCKET = ['lt10', '10to50', 'gt50'] as const;
const THIRD = ['bottom', 'middle', 'top'] as const;
const EQ_KIND = ['any', 'highs', 'lows'] as const;
const SESSION = ['asia', 'london', 'new_york', 'overlap'] as const;
const BARS_RECENCY = [5, 10, 20, 50] as const;
const BARS_FORMED = [10, 20, 50] as const;
const PROX_ZONE = [0.1, 0.25, 0.5] as const;
const PROX_LIQ = [0.25, 0.5, 1] as const;

function ctrl(
  name: ControlName,
  values: readonly (string | number)[],
  def: string | number,
) {
  return { name, values: [...values], default: def };
}

export const CONDITION_PALETTE: readonly PaletteEntry[] = [
  // ── STRUCTURE ──────────────────────────────────────────────────────────────
  {
    type: 'trend_is',
    family: 'structure',
    label: 'La tendance structurelle est',
    description:
      'La tendance de STRUCTURE sur ce timeframe (dernière cassure BOS/CHOCH non contredite) est, en ce moment, celle choisie — haussière, baissière, ou indéterminée quand aucune cassure ne l’a établie.',
    controls: [ctrl('trend', TREND, 'bullish')],
    tense: 'present',
  },
  {
    type: 'mtf_aligned',
    family: 'structure',
    label: 'Les 3 unités H4/H1/M15 s’accordent (structure)',
    description:
      'Les trois timeframes H4, H1 et M15 montrent la MÊME direction de structure en ce moment. Un timeframe sans tendance structurelle établie (indéterminé) empêche l’accord. Condition transitoire : elle compare trois unités fixes, non « l’unité au-dessus » de celle scannée.',
    controls: [ctrl('direction', DIRECTION, 'any')],
    tense: 'present',
    supports_direction: true,
  },
  {
    type: 'last_event_is',
    family: 'structure',
    label: 'Le dernier événement détecté est',
    description:
      'Le tout dernier événement de structure daté (le plus récent entre BOS et CHOCH, haussier ou baissier) est celui choisi.',
    controls: [ctrl('event', EVENT, 'bos_up')],
    tense: 'present',
  },
  {
    type: 'bos_recent_confirmed',
    family: 'structure',
    label: 'Un BOS est survenu dans les N dernières bougies',
    description:
      'Une cassure de structure (BOS) confirmée est datée d’au plus N bougies — une distance comptée en bougies, jamais en minutes.',
    controls: [ctrl('direction', DIRECTION, 'any'), ctrl('max_bars', BARS_RECENCY, 10)],
    tense: 'present',
    supports_direction: true,
  },
  {
    type: 'choch_recent_confirmed',
    family: 'structure',
    label: 'Un CHOCH est survenu dans les N dernières bougies',
    description:
      'Un changement de caractère (CHOCH) confirmé est daté d’au plus N bougies — une distance comptée en bougies, jamais en minutes.',
    controls: [ctrl('direction', DIRECTION, 'any'), ctrl('max_bars', BARS_RECENCY, 10)],
    tense: 'present',
    supports_direction: true,
  },
  {
    type: 'last_event_age',
    family: 'structure',
    label: 'Le dernier événement remonte à',
    description:
      'Le dernier événement de structure daté remonte à moins de 10 bougies, entre 10 et 50, ou plus de 50 — compté en bougies.',
    controls: [ctrl('age_bucket', AGE_BUCKET, 'lt10')],
    tense: 'present',
  },
  // ── ZONES ──────────────────────────────────────────────────────────────────
  {
    type: 'price_in_ob',
    family: 'zones',
    label: 'Le prix est dans un Order Block',
    description: 'Le prix courant se situe à l’intérieur d’un Order Block actif.',
    controls: [ctrl('direction', DIRECTION, 'any')],
    tense: 'present',
    supports_direction: true,
  },
  {
    type: 'price_in_fvg',
    family: 'zones',
    label: 'Le prix est dans un Fair Value Gap',
    description: 'Le prix courant se situe à l’intérieur d’un Fair Value Gap non comblé.',
    controls: [ctrl('direction', DIRECTION, 'any')],
    tense: 'present',
    supports_direction: true,
  },
  {
    type: 'price_in_tested_zone',
    family: 'zones',
    label: 'Le prix est dans une zone déjà testée',
    description:
      'Le prix courant est à l’intérieur d’une zone active (Order Block ou Fair Value Gap) qui a déjà été testée au moins une fois depuis sa formation.',
    controls: [ctrl('zone_kind', ZONE_KIND, 'any')],
    tense: 'present',
  },
  {
    type: 'zone_untested',
    family: 'zones',
    label: 'Une zone n’a jamais été testée depuis sa formation',
    description:
      'Il existe une zone active (Order Block ou Fair Value Gap) qui n’a jamais été retestée depuis sa formation.',
    controls: [ctrl('zone_kind', ZONE_KIND, 'any')],
    tense: 'present',
  },
  {
    type: 'zone_formed_recent',
    family: 'zones',
    label: 'Une zone s’est formée dans les N dernières bougies',
    description:
      'Il existe une zone active (Order Block ou Fair Value Gap) formée il y a au plus N bougies — compté en bougies.',
    controls: [ctrl('zone_kind', ZONE_KIND, 'any'), ctrl('max_bars', BARS_FORMED, 20)],
    tense: 'present',
  },
  {
    type: 'price_near_ob',
    family: 'zones',
    label: 'Le prix est à moins de X % d’un Order Block, sans y être',
    description:
      'Le prix courant est à moins de la distance choisie d’un Order Block actif, SANS être à l’intérieur.',
    controls: [ctrl('direction', DIRECTION, 'any'), ctrl('proximity_pct', PROX_ZONE, 0.25)],
    tense: 'present',
    supports_direction: true,
  },
  {
    type: 'price_near_fvg',
    family: 'zones',
    label: 'Le prix est à moins de X % d’un Fair Value Gap, sans y être',
    description:
      'Le prix courant est à moins de la distance choisie d’un Fair Value Gap non comblé, SANS être à l’intérieur.',
    controls: [ctrl('direction', DIRECTION, 'any'), ctrl('proximity_pct', PROX_ZONE, 0.25)],
    tense: 'present',
    supports_direction: true,
  },
  // ── LIQUIDITY ──────────────────────────────────────────────────────────────
  {
    type: 'price_near_liquidity',
    family: 'liquidity',
    label: 'Une poche intacte est à moins de X % du prix',
    description:
      'Le prix courant est à moins de la distance choisie d’une poche de liquidité INTACTE (BSL au-dessus / SSL en dessous).',
    controls: [ctrl('side', SIDE, 'any'), ctrl('proximity_pct', PROX_LIQ, 0.5)],
    tense: 'present',
  },
  {
    type: 'liquidity_swept_recent',
    family: 'liquidity',
    label: 'Une poche a été prise dans les N dernières bougies',
    description:
      'Une poche de liquidité (SSL/BSL) a été balayée — sa liquidité prise — il y a au plus N bougies. Compté en bougies.',
    controls: [ctrl('side', SIDE, 'any'), ctrl('max_bars', BARS_RECENCY, 10)],
    tense: 'present',
  },
  {
    type: 'liquidity_broken_recent',
    family: 'liquidity',
    label: 'Une poche a été cassée dans les N dernières bougies',
    description:
      'Une poche de liquidité (SSL/BSL) a été cassée nettement — clôture au travers, état terminal — il y a au plus N bougies. À distinguer d’une simple prise (balayage). Compté en bougies.',
    controls: [ctrl('side', SIDE, 'any'), ctrl('max_bars', BARS_RECENCY, 10)],
    tense: 'present',
  },
  {
    type: 'equal_levels_present',
    family: 'liquidity',
    label: 'Des creux ou sommets égaux sont présents',
    description:
      'Au moins une poche de liquidité intacte formée de creux égaux (EQL) ou de sommets égaux (EQH) est présente.',
    controls: [ctrl('eq_kind', EQ_KIND, 'any')],
    tense: 'present',
  },
  // ── CONTEXT ────────────────────────────────────────────────────────────────
  {
    type: 'market_phase_is',
    family: 'context',
    label: 'La phase de marché est',
    description:
      'La phase de marché observée correspond à celle choisie — accumulation, tendance, range ou expansion.',
    controls: [ctrl('phase', PHASE, 'expansion')],
    tense: 'present',
  },
  {
    type: 'volatility_is',
    family: 'context',
    label: 'La volatilité est',
    description:
      'La volatilité observée en ce moment correspond au niveau choisi — contractée (faible), normale, ou étendue (élevée).',
    controls: [ctrl('volatility', VOLATILITY, 'elevated')],
    tense: 'present',
  },
  {
    type: 'price_in_range_third',
    family: 'context',
    label: 'Le prix est dans le tiers … du range',
    description:
      'Le prix courant se situe dans le tiers haut, milieu ou bas du range STRUCTUREL de la fenêtre d’analyse (mêmes bornes que la tuile « Position dans le range » du panneau Régime).',
    controls: [ctrl('third', THIRD, 'bottom')],
    tense: 'present',
  },
  {
    type: 'session_is',
    family: 'context',
    label: 'La session en cours est',
    description:
      'Au moment de la dernière bougie clôturée, la session de marché est celle choisie — Asie, Londres, New York, ou le chevauchement Londres/New York.',
    controls: [ctrl('session', SESSION, 'london')],
    tense: 'present',
  },
] as const;

export const CONDITION_TYPES: readonly ConditionType[] = CONDITION_PALETTE.map((p) => p.type);

export function paletteEntry(type: ConditionType): PaletteEntry | undefined {
  return CONDITION_PALETTE.find((p) => p.type === type);
}

/** Group a palette into its four families, in fixed order. */
export function groupByFamily(
  palette: readonly PaletteEntry[],
): Array<{ family: Family; entries: PaletteEntry[] }> {
  return FAMILIES.map((family) => ({
    family,
    entries: palette.filter((p) => p.family === family),
  }));
}

/**
 * French fallback labels for each segmented option value. The builder localises
 * via i18n first (scanner.opt.<control>.<value>) and falls back to these, so a
 * value added before its translation still reads in French, never as a raw enum.
 */
export const OPTION_LABELS_FR: Record<ControlName, Record<string, string>> = {
  direction: { any: 'Toute direction', bullish: 'Haussier', bearish: 'Baissier' },
  trend: { bullish: 'Haussière', bearish: 'Baissière', indeterminate: 'Indéterminée' },
  phase: { accumulation: 'Accumulation', trend: 'Tendance', ranging: 'Range', expansion: 'Expansion' },
  volatility: { low: 'Contractée', normal: 'Normale', elevated: 'Étendue' },
  side: { any: 'Les deux', bsl: 'BSL (au-dessus)', ssl: 'SSL (en dessous)' },
  zone_kind: { any: 'OB ou FVG', ob: 'Order Block', fvg: 'Fair Value Gap' },
  event: { bos_up: 'BOS ↑', bos_down: 'BOS ↓', choch_up: 'CHOCH ↑', choch_down: 'CHOCH ↓' },
  age_bucket: { lt10: 'Moins de 10', '10to50': '10 à 50', gt50: 'Plus de 50' },
  third: { bottom: 'Bas', middle: 'Milieu', top: 'Haut' },
  eq_kind: { any: 'Creux ou sommets', highs: 'Sommets', lows: 'Creux' },
  session: { asia: 'Asie', london: 'Londres', new_york: 'New York', overlap: 'Chevauchement' },
  max_bars: {},
  proximity_pct: {},
};

/** Human label for a segmented value (numbers stay factual: bars / %). */
export function optionLabelFallback(control: ControlName, value: string | number): string {
  if (control === 'max_bars') return `${value} bougies`;
  if (control === 'proximity_pct') return `${value} %`;
  return OPTION_LABELS_FR[control]?.[String(value)] ?? String(value);
}
