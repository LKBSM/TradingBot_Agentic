'use client';

import { useTranslations } from 'next-intl';

/**
 * Locale-aware display helpers for the Scanner. Replaces the former pure
 * `biasLabel` / `phaseLabel` / `relativeAge` in `labels.ts` — those produced
 * hard-coded French. The colour/tone/glyph helpers stay pure in `labels.ts`
 * (they carry no translatable text).
 *
 * All scanner consumers are client components, so a hook reading
 * `useTranslations('scanner')` is the clean seam.
 */
export function useScannerLabels() {
  const t = useTranslations('scanner');

  /** Bias label for a market-context trend value (ComboCard). */
  function bias(value: string | null | undefined): string {
    switch (value) {
      case 'bullish':
        return t('bias.bullish');
      case 'bearish':
        return t('bias.bearish');
      case 'ranging':
        return t('bias.ranging');
      case 'neutral':
        return t('bias.neutral');
      default:
        return t('bias.unknown');
    }
  }

  /** Market-phase label (ComboCard context badge). */
  function phase(value: string | null | undefined): string {
    switch (value) {
      case 'accumulation':
        return t('phase.accumulation');
      case 'distribution':
        return t('phase.distribution');
      case 'trend':
        return t('phase.trend');
      case 'ranging':
        return t('phase.ranging');
      case 'expansion':
        return t('phase.expansion');
      default:
        return t('phase.unknown');
    }
  }

  /**
   * Localized relative age from an ISO timestamp ("à l'instant", "il y a 3 min",
   * "il y a 2 h", "il y a 4 j"). Returns null for a missing/invalid timestamp so
   * callers can render their own empty state. Buckets mirror the former
   * `relativeAge` / `freshnessLabel` so existing behaviour (and tests) hold.
   */
  function age(iso: string | null | undefined, nowMs: number): string | null {
    if (!iso) return null;
    const then = Date.parse(iso);
    if (Number.isNaN(then)) return null;
    const minutes = Math.max(0, Math.round((nowMs - then) / 60000));
    if (minutes < 1) return t('age.instant');
    if (minutes < 60) return t('age.min', { n: minutes });
    const hours = Math.round(minutes / 60);
    if (hours < 48) return t('age.hours', { n: hours });
    const days = Math.round(hours / 24);
    return t('age.days', { n: days });
  }

  return { bias, phase, age };
}
