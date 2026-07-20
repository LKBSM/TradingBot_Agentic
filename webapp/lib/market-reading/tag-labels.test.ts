import { describe, expect, it } from 'vitest';
import { humaniseTag } from './tag-labels';
import fr from '@/messages/fr.json';

// Tag labels moved to the `reading.tags.*` message namespace (i18n Étape 3);
// `formatTag` is gone. The invariant it protected — every backend tag maps to a
// real FR label, no English snake_case leak — is now asserted against fr.json.
const TAGS: Record<string, string> = fr.reading.tags;

describe('humaniseTag (unmapped-tag fallback)', () => {
  it('never leaks a raw snake_case code for an unknown tag', () => {
    const out = humaniseTag('some_unknown_code');
    expect(out).not.toContain('_');
    expect(out).toBe('Some unknown code');
  });
});

describe('reading.tags catalog (FR)', () => {
  it('maps known snake_case codes to plain-language labels', () => {
    expect(TAGS['trend']).toBe('Tendance établie');
    expect(TAGS['bos_confirmed']).toBe('Cassure confirmée');
    expect(TAGS['retest_active']).toBe('Retest en cours');
    expect(TAGS['low_vol']).toBe('Volatilité basse');
    expect(TAGS['choch_pending']).toBe('Changement de caractère en attente');
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
    for (const tag of backendTags) {
      const label = TAGS[tag];
      expect(label, `${tag} is mapped`).toBeTruthy();
      // A mapped tag must NOT equal its naive snake_case fallback (= English leak).
      expect(label, `${tag} is not the English fallback`).not.toBe(humaniseTag(tag));
      expect(label).not.toContain('_');
    }
    // Spot-check the exact strings the founder reported.
    expect(TAGS['trend_bearish']).toBe('Tendance baissière');
    expect(TAGS['volatility_elevated']).toBe('Volatilité élevée');
    expect(TAGS['phase_expansion']).toBe('Phase d’expansion');
    expect(TAGS['retest_in_progress']).toBe('Retest en cours');
  });
});
