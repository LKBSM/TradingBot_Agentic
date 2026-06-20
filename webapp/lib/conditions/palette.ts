/**
 * Conditions Scanner — the palette (single source of truth for the builder).
 *
 * Every entry is a STRUCTURAL FACT AT THE PRESENT. There is intentionally no
 * predictive / outcome condition ("will bounce", "will break"): such a thing is
 * not representable here, and a test asserts the palette stays present-tense.
 *
 * Mirrors the backend palette (src/intelligence/conditions_scanner.py:PALETTE).
 */

import type { ConditionType, PaletteEntry } from './types';

export const CONDITION_PALETTE: readonly PaletteEntry[] = [
  {
    type: 'mtf_aligned',
    label: '3 TF alignés',
    description:
      'Les 3 timeframes (H4, H1, M15) pointent dans la même direction en ce moment.',
    supportsDirection: true,
    tense: 'present',
  },
  {
    type: 'price_in_ob',
    label: 'Prix dans un Order Block',
    description: 'Le prix courant se situe à l’intérieur d’un Order Block actif.',
    supportsDirection: true,
    tense: 'present',
  },
  {
    type: 'price_in_fvg',
    label: 'Prix dans un Fair Value Gap',
    description:
      'Le prix courant se situe à l’intérieur d’un Fair Value Gap non comblé.',
    supportsDirection: true,
    tense: 'present',
  },
  {
    type: 'ob_fvg_confluence',
    label: 'Confluence OB + FVG au prix courant',
    description:
      'Le prix courant se situe simultanément dans un Order Block actif et dans un Fair Value Gap non comblé.',
    supportsDirection: false,
    tense: 'present',
  },
  {
    type: 'bos_recent_confirmed',
    label: 'BOS confirmé récent',
    description:
      'Une cassure de structure (BOS) confirmée est datée des dernières bougies.',
    supportsDirection: true,
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

export const DEFAULT_BOS_MAX_BARS = 5;
