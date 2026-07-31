import { describe, expect, it } from 'vitest';
import { CONDITION_PALETTE, CONDITION_TYPES, FAMILIES, groupByFamily } from '../palette';

const EXPECTED_TYPES = new Set([
  // structure
  'trend_is', 'higher_tf_agrees', 'last_event_is', 'last_event_age',
  'bos_recent_confirmed', 'choch_recent_confirmed',
  // zones
  'price_in_ob', 'price_in_fvg', 'price_in_tested_zone', 'zone_untested',
  'zone_tested_at_most', 'zone_formed_recent', 'price_near_ob', 'price_near_fvg',
  // liquidity
  'price_near_liquidity', 'liquidity_swept_recent', 'liquidity_broken_recent', 'equal_levels_present',
  // context
  'market_phase_is', 'volatility_is', 'price_in_range_third', 'session_is',
]);

describe('conditions palette (SC-1)', () => {
  it('exposes exactly the SC-1 present-tense condition types', () => {
    expect(new Set(CONDITION_TYPES)).toEqual(EXPECTED_TYPES);
  });

  it('no longer offers the removed conditions', () => {
    expect(CONDITION_TYPES).not.toContain('ob_fvg_confluence');
    expect(CONDITION_TYPES).not.toContain('retest_in_progress');
    expect(CONDITION_TYPES).not.toContain('mtf_aligned');
  });

  it('groups every entry into one of the four families, with controls', () => {
    for (const entry of CONDITION_PALETTE) {
      expect(FAMILIES).toContain(entry.family);
      expect(entry.tense).toBe('present');
      expect(Array.isArray(entry.controls)).toBe(true);
      for (const c of entry.controls) {
        expect(c.values.length).toBeGreaterThan(0);
        expect(c.default).toBeDefined();
      }
    }
  });

  it('groupByFamily returns the four families in fixed order, covering all entries', () => {
    const groups = groupByFamily(CONDITION_PALETTE);
    expect(groups.map((g) => g.family)).toEqual(['structure', 'zones', 'liquidity', 'context']);
    expect(groups.reduce((n, g) => n + g.entries.length, 0)).toBe(CONDITION_PALETTE.length);
  });

  it('excludes the unreachable market phase (distribution)', () => {
    const phase = CONDITION_PALETTE.find((p) => p.type === 'market_phase_is');
    expect(phase?.controls[0]?.values).not.toContain('distribution');
  });

  it('offers no predictive / outcome vocabulary (descriptive only)', () => {
    const forbidden = [
      'rebond', 'cassera', 'va casser', 'va rebondir', 'prédi', 'predict', 'probab',
      'cible', 'target', 'gagnant', 'prévision', 'meilleur', 'score', 'continuera',
      'renvers', 'idéal', 'recommand', 'opportun', 'setup', 'plus sûr', 'qualité',
    ];
    for (const entry of CONDITION_PALETTE) {
      const haystack = `${entry.type} ${entry.label} ${entry.description}`.toLowerCase();
      for (const word of forbidden) {
        expect(haystack, `palette entry ${entry.type}`).not.toContain(word);
      }
    }
  });
});
