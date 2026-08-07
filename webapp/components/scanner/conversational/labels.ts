'use client';

import { useTranslations } from 'next-intl';
import type { ConditionType, ControlName, Family } from '@/lib/conditions/types';
import { optionLabelFallback, paletteEntry } from '@/lib/conditions/palette';

/**
 * Localised labels for the conversational scanner, sourced from the SAME
 * `scanner` i18n namespace the classic builder uses (already translated across
 * the 9 locales). Palette/option vocabulary therefore stays a single source; the
 * conversational surface only adds its own chrome under `scannerChat`.
 */
export function useConditionLabels() {
  const t = useTranslations('scanner');

  const conditionLabel = (type: ConditionType): string =>
    t.has(`palette.${type}_label`) ? t(`palette.${type}_label`) : paletteEntry(type)?.label ?? type;

  const optionLabel = (control: ControlName, value: string | number): string => {
    if (control === 'max_bars') return t('opt.bars', { n: value });
    if (control === 'proximity_pct') return t('opt.pct', { n: value });
    if (control === 'max_touches') return t('opt.touches', { n: value });
    const key = `opt.${control}.${value}`;
    return t.has(key) ? t(key) : optionLabelFallback(control, value);
  };

  const familyLabel = (family: Family): string =>
    t.has(`family.${family}.title`) ? t(`family.${family}.title`) : family;

  return { conditionLabel, optionLabel, familyLabel };
}
