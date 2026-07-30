import { describe, expect, it } from 'vitest';
import {
  classifyMtfAlignment,
  describeMtfAlignment,
  mtfOrderFor,
  mtfTrendGlyph,
  type MtfEntry,
  type MtfTrendMap,
} from '../mtf-trend';

// A sample 3-unit order (as if viewing M15 → its upper units). Kept explicit so
// the descriptive assertions are stable regardless of the registry ladder.
const ORDER: MtfEntry[] = [
  { key: 'h1', label: 'H1', tf: 'H1' },
  { key: 'h4', label: 'H4', tf: 'H4' },
  { key: 'd1', label: 'D1', tf: 'D1' },
];
const map = (
  h1: MtfTrendMap[string],
  h4: MtfTrendMap[string],
  d1: MtfTrendMap[string],
): MtfTrendMap => ({ h1, h4, d1 });

describe('mtfOrderFor — units ABOVE the viewed one (TF-1 decision C)', () => {
  it('is relative to the viewed timeframe', () => {
    expect(mtfOrderFor('M5').map((e) => e.tf)).toEqual(['M15', 'H1', 'H4', 'D1']);
    expect(mtfOrderFor('H1').map((e) => e.tf)).toEqual(['H4', 'D1']);
    expect(mtfOrderFor('H4').map((e) => e.tf)).toEqual(['D1']);
    expect(mtfOrderFor('D1').map((e) => e.tf)).toEqual(['W1']); // top of perimeter → reference
    expect(mtfOrderFor('W1').map((e) => e.tf)).toEqual([]); // nothing above
  });
});

describe('mtfTrendGlyph', () => {
  it('maps each trend to a descriptive arrow + tone', () => {
    expect(mtfTrendGlyph('bullish')).toEqual({ arrow: '↗', tone: 'bull' });
    expect(mtfTrendGlyph('bearish')).toEqual({ arrow: '↘', tone: 'bear' });
    expect(mtfTrendGlyph('indeterminate')).toEqual({ arrow: '–', tone: 'neutral' });
    expect(mtfTrendGlyph(null)).toEqual({ arrow: '·', tone: 'neutral' });
  });
});

describe('describeMtfAlignment', () => {
  it('all bullish → aligned (haussières), named "unités supérieures"', () => {
    expect(describeMtfAlignment(map('bullish', 'bullish', 'bullish'), ORDER)).toBe(
      'Les 3 unités supérieures sont haussières.',
    );
  });

  it('all bearish → aligned (baissières)', () => {
    expect(describeMtfAlignment(map('bearish', 'bearish', 'bearish'), ORDER)).toBe(
      'Les 3 unités supérieures sont baissières.',
    );
  });

  it('indeterminate treated as flat → all sans tendance structurelle établie', () => {
    expect(describeMtfAlignment(map('indeterminate', 'indeterminate', 'indeterminate'), ORDER)).toBe(
      'Les 3 unités supérieures sont sans tendance structurelle établie.',
    );
  });

  it('a single available upper unit uses the singular', () => {
    expect(describeMtfAlignment(map('bullish', null, null), ORDER)).toBe(
      "L'unité supérieure est haussière.",
    );
  });

  it('mixed directions → named listing in order', () => {
    expect(describeMtfAlignment(map('bullish', 'indeterminate', 'bearish'), ORDER)).toBe(
      'Unités supérieures : H1 haussier, H4 indéterminé et D1 baissier.',
    );
  });

  it('counts only the available units (partial reads)', () => {
    expect(describeMtfAlignment(map('bullish', 'bullish', null), ORDER)).toBe(
      'Les 2 unités supérieures sont haussières.',
    );
  });

  it('empty string when nothing is available', () => {
    expect(describeMtfAlignment(map(null, null, null), ORDER)).toBe('');
    expect(describeMtfAlignment({}, [])).toBe('');
  });

  it('never emits predictive / score / action vocabulary', () => {
    const samples = [
      describeMtfAlignment(map('bullish', 'bullish', 'bullish'), ORDER),
      describeMtfAlignment(map('bullish', 'bullish', 'bearish'), ORDER),
      describeMtfAlignment(map('bullish', 'indeterminate', 'bearish'), ORDER),
      describeMtfAlignment(map('indeterminate', 'indeterminate', 'indeterminate'), ORDER),
    ].join(' ');
    const forbidden = [/va\s/i, /attends/i, /\d+\s*%/, /setup/i, /fort/i, /évite/i, /risqu/i, /probab/i];
    for (const re of forbidden) expect(samples).not.toMatch(re);
  });
});

describe('classifyMtfAlignment (relation kind + disagreement flag)', () => {
  it('all aligned → kind aligned, no disagreement', () => {
    const r = classifyMtfAlignment(map('bullish', 'bullish', 'bullish'), ORDER);
    expect(r.kind).toBe('aligned');
    expect(r.disagreement).toBe(false);
  });

  it('all flat → kind neutral, no disagreement', () => {
    const r = classifyMtfAlignment(map('indeterminate', 'indeterminate', 'indeterminate'), ORDER);
    expect(r.kind).toBe('neutral');
    expect(r.disagreement).toBe(false);
  });

  it('up AND down both present → kind divergent, DISAGREEMENT', () => {
    const r = classifyMtfAlignment(map('bullish', 'indeterminate', 'bearish'), ORDER);
    expect(r.kind).toBe('divergent');
    expect(r.disagreement).toBe(true);
  });

  it('one direction + flats (no opposite) → kind partial, NO disagreement', () => {
    const r = classifyMtfAlignment(map('bullish', 'indeterminate', 'bullish'), ORDER);
    expect(r.kind).toBe('partial');
    expect(r.disagreement).toBe(false);
  });

  it('nothing available → kind none, empty text', () => {
    const r = classifyMtfAlignment(map(null, null, null), ORDER);
    expect(r.kind).toBe('none');
    expect(r.text).toBe('');
    expect(r.disagreement).toBe(false);
  });
});
