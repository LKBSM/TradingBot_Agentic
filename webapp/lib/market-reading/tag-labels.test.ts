import { describe, expect, it } from 'vitest';
import { formatTag } from './tag-labels';

describe('formatTag (vulgarisation des tags — 5.D)', () => {
  it('maps known snake_case codes to plain-language labels', () => {
    expect(formatTag('trend')).toBe('Tendance établie');
    expect(formatTag('bos_confirmed')).toBe('Cassure confirmée');
    expect(formatTag('retest_active')).toBe('Retest en cours');
    expect(formatTag('low_vol')).toBe('Volatilité basse');
    expect(formatTag('choch_pending')).toBe('Changement de caractère en attente');
  });

  it('never leaks a raw snake_case code for an unknown tag', () => {
    const out = formatTag('some_unknown_code');
    expect(out).not.toContain('_');
    expect(out).toBe('Some unknown code');
  });

  // Regression — i18n leak (eval founder 2026-06-08): the backend emits
  // `trend_<v>` / `volatility_<v>` / `phase_<v>` / …; without a mapping they fell
  // back to humanise() and surfaced English-looking labels in FR ("Trend
  // bearish", "Volatility elevated", "Phase expansion", "Retest in progress").
  it('maps every tag the backend actually emits to a French label (no English leak)', () => {
    const backendTags = [
      'trend_bullish',
      'trend_bearish',
      'trend_neutral',
      'trend_ranging',
      'volatility_low',
      'volatility_normal',
      'volatility_elevated',
      'phase_accumulation',
      'phase_distribution',
      'phase_trend',
      'phase_ranging',
      'phase_expansion',
      'bos_recent_bullish',
      'bos_recent_bearish',
      'choch_recent_bullish',
      'choch_recent_bearish',
      'retest_in_progress',
      'ob_active',
      'fvg_active',
      'mtf_aligned',
      'mtf_divergent',
      'mtf_mixed',
    ];
    // The naive snake_case fallback ("trend_bearish" → "Trend bearish") is the
    // English leak. A mapped tag must NOT equal its fallback form.
    const humanised = (t: string) => {
      const s = t.replace(/_/g, ' ').trim();
      return s.charAt(0).toUpperCase() + s.slice(1);
    };
    for (const tag of backendTags) {
      const label = formatTag(tag);
      expect(label, `${tag} is mapped (not the English fallback)`).not.toBe(
        humanised(tag),
      );
      expect(label).not.toContain('_');
    }
    // Spot-check the exact strings the founder reported.
    expect(formatTag('trend_bearish')).toBe('Tendance baissière');
    expect(formatTag('volatility_elevated')).toBe('Volatilité élevée');
    expect(formatTag('phase_expansion')).toBe('Phase d’expansion');
    expect(formatTag('retest_in_progress')).toBe('Retest en cours');
  });
});
