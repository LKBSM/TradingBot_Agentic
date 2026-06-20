import { describe, expect, it } from 'vitest';
import { CONDITION_PALETTE, CONDITION_TYPES } from '../palette';

describe('conditions palette', () => {
  it('exposes exactly the 10 present-tense condition types', () => {
    expect(CONDITION_PALETTE).toHaveLength(10);
    expect(new Set(CONDITION_TYPES)).toEqual(
      new Set([
        'mtf_aligned',
        'trend_is',
        'market_phase_is',
        'volatility_is',
        'price_in_ob',
        'price_in_fvg',
        'ob_fvg_confluence',
        'bos_recent_confirmed',
        'choch_recent_confirmed',
        'retest_in_progress',
      ]),
    );
  });

  it('marks every entry as present-tense', () => {
    for (const entry of CONDITION_PALETTE) {
      expect(entry.tense).toBe('present');
    }
  });

  it('offers no predictive / outcome condition (descriptive only)', () => {
    const forbidden = [
      'rebond',
      'cassera',
      'va casser',
      'va rebondir',
      'prédi',
      'predict',
      'probab',
      'cible',
      'target',
      'gagnant',
      'prévision',
      'meilleur',
      'score',
      'continuera',
      'renvers',
    ];
    for (const entry of CONDITION_PALETTE) {
      const haystack = `${entry.type} ${entry.label} ${entry.description}`.toLowerCase();
      for (const word of forbidden) {
        expect(haystack, `palette entry ${entry.type}`).not.toContain(word);
      }
    }
  });
});
