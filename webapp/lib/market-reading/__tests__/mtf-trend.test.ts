import { describe, expect, it } from 'vitest';
import {
  describeMtfAlignment,
  mtfTrendGlyph,
  type MtfTrendMap,
} from '../mtf-trend';

const map = (
  h4: MtfTrendMap['h4'],
  h1: MtfTrendMap['h1'],
  m15: MtfTrendMap['m15'],
): MtfTrendMap => ({ h4, h1, m15 });

describe('mtfTrendGlyph', () => {
  it('maps each trend to a descriptive arrow + tone', () => {
    expect(mtfTrendGlyph('bullish')).toEqual({ arrow: '↗', tone: 'bull' });
    expect(mtfTrendGlyph('bearish')).toEqual({ arrow: '↘', tone: 'bear' });
    expect(mtfTrendGlyph('neutral')).toEqual({ arrow: '→', tone: 'neutral' });
    expect(mtfTrendGlyph('ranging')).toEqual({ arrow: '→', tone: 'neutral' });
    expect(mtfTrendGlyph(null)).toEqual({ arrow: '·', tone: 'neutral' });
  });
});

describe('describeMtfAlignment', () => {
  it('all three bullish → aligned (haussiers)', () => {
    expect(describeMtfAlignment(map('bullish', 'bullish', 'bullish'))).toBe(
      'Les 3 TF sont alignés (haussiers).',
    );
  });

  it('all three bearish → aligned (baissiers)', () => {
    expect(describeMtfAlignment(map('bearish', 'bearish', 'bearish'))).toBe(
      'Les 3 TF sont alignés (baissiers).',
    );
  });

  it('treats ranging/neutral as flat → all neutres', () => {
    expect(describeMtfAlignment(map('neutral', 'ranging', 'neutral'))).toBe(
      'Les 3 TF sont neutres.',
    );
  });

  it('higher TFs agree, M15 opposes → pullback line', () => {
    expect(describeMtfAlignment(map('bullish', 'bullish', 'bearish'))).toBe(
      'M15 se replie contre la tendance H4 haussière.',
    );
    expect(describeMtfAlignment(map('bearish', 'bearish', 'bullish'))).toBe(
      'M15 se replie contre la tendance H4 baissière.',
    );
  });

  it('mixed directions → divergence listing in H4→M15 order', () => {
    expect(describeMtfAlignment(map('bullish', 'neutral', 'bearish'))).toBe(
      'Les TF divergent : H4 haussier, H1 neutre et M15 baissier.',
    );
  });

  it('describes only the available timeframes (partial reads)', () => {
    // Only H1 and H4 available, both bullish → aligned, count = 2.
    expect(describeMtfAlignment(map('bullish', 'bullish', null))).toBe(
      'Les 2 TF sont alignés (haussiers).',
    );
    // A single available, non-aligned set falls to divergence listing.
    expect(describeMtfAlignment(map('bullish', null, 'bearish'))).toBe(
      'Les TF divergent : H4 haussier et M15 baissier.',
    );
  });

  it('returns empty string when nothing is available', () => {
    expect(describeMtfAlignment(map(null, null, null))).toBe('');
  });

  it('never emits predictive / score / action vocabulary', () => {
    const samples = [
      describeMtfAlignment(map('bullish', 'bullish', 'bullish')),
      describeMtfAlignment(map('bullish', 'bullish', 'bearish')),
      describeMtfAlignment(map('bullish', 'neutral', 'bearish')),
      describeMtfAlignment(map('neutral', 'neutral', 'neutral')),
    ].join(' ');
    const forbidden = [
      /va\s/i,
      /tendance à/i,
      /attends/i,
      /\d+\s*%/,
      /setup/i,
      /fort/i,
      /évite/i,
      /risqu/i,
      /bon moment/i,
      /probab/i,
    ];
    for (const re of forbidden) expect(samples).not.toMatch(re);
  });
});
