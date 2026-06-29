/**
 * Pure helpers for the multi-timeframe TREND ALIGNMENT panel.
 *
 * These functions only CLASSIFY trend values the engine already produced for
 * each timeframe (read from each timeframe's existing market reading). They
 * never compute or recompute a trend, never score, never predict. Every string
 * is present-tense and strictly descriptive — it states what IS observed across
 * M15 / H1 / H4, nothing about what will happen.
 */
import type { TrendValue } from '@/types/market-reading';
import type { Tone } from './formatters';

/** The three timeframes summarised, highest → lowest (display order). */
export const MTF_TREND_ORDER = [
  { key: 'h4', label: 'H4', tf: 'H4' },
  { key: 'h1', label: 'H1', tf: 'H1' },
  { key: 'm15', label: 'M15', tf: 'M15' },
] as const;

export type MtfKey = (typeof MTF_TREND_ORDER)[number]['key'];

/** Current trend per timeframe; null when that timeframe's read is unavailable. */
export type MtfTrendMap = Record<MtfKey, TrendValue | null>;

/**
 * Arrow glyph + tone for a single timeframe's trend. Descriptive only:
 * bullish ↗, bearish ↘, neutral/ranging →, unavailable ·.
 */
export function mtfTrendGlyph(trend: TrendValue | null): {
  arrow: string;
  tone: Tone;
} {
  switch (trend) {
    case 'bullish':
      return { arrow: '↗', tone: 'bull' };
    case 'bearish':
      return { arrow: '↘', tone: 'bear' };
    case 'neutral':
    case 'ranging':
      return { arrow: '→', tone: 'neutral' };
    default:
      return { arrow: '·', tone: 'neutral' };
  }
}

const TREND_ADJ: Record<TrendValue, string> = {
  bullish: 'haussier',
  bearish: 'baissier',
  neutral: 'neutre',
  ranging: 'en range',
};

type Dir = 'up' | 'down' | 'flat';
const dirOf = (t: TrendValue): Dir =>
  t === 'bullish' ? 'up' : t === 'bearish' ? 'down' : 'flat';

/** French enumeration: ["a","b","c"] → "a, b et c". */
function joinFr(parts: string[]): string {
  if (parts.length <= 1) return parts.join('');
  return `${parts.slice(0, -1).join(', ')} et ${parts[parts.length - 1]}`;
}

/**
 * Relation between the timeframes — the classification underlying both the
 * descriptive line and the multi-TF DISAGREEMENT callout.
 */
export interface MtfRelation {
  /**
   *   · 'aligned'    all available TFs share one non-flat direction
   *   · 'neutral'    all available TFs are flat (neutral / ranging)
   *   · 'pullback'   the higher TFs agree and M15 takes the opposite direction
   *   · 'divergent'  both an up and a down direction are present (contradiction)
   *   · 'partial'    a mix of one direction and flats (no opposite direction)
   *   · 'none'       no timeframe available
   */
  kind: 'aligned' | 'neutral' | 'pullback' | 'divergent' | 'partial' | 'none';
  /** Present-tense description; '' when kind === 'none'. */
  text: string;
  /**
   * True only when a timeframe genuinely goes AGAINST another — i.e. an up and a
   * down direction coexist (pullback or divergent). A direction-vs-flat mix is
   * NOT a disagreement. Drives the warn callout so the « contre » is as readable
   * as the « accord ».
   */
  disagreement: boolean;
}

/**
 * Classify the RELATION between the timeframes, derived purely from the already-
 * computed trend values:
 *   · all same non-flat  → "Les 3 TF sont alignés (haussiers)."
 *   · all flat           → "Les 3 TF sont neutres."
 *   · H4+H1 agree, M15 opposes → "M15 se replie contre la tendance H4 haussière."
 *   · otherwise          → "Les TF divergent : H4 haussier, H1 neutre et M15 baissier."
 * Strictly descriptive — no future tense, no probability, no score, no action verdict.
 */
export function classifyMtfAlignment(trends: MtfTrendMap): MtfRelation {
  const entries = MTF_TREND_ORDER.map(({ key, label }) => ({
    label: label as string,
    trend: trends[key],
  })).filter(
    (e): e is { label: string; trend: TrendValue } => e.trend != null,
  );

  if (entries.length === 0) return { kind: 'none', text: '', disagreement: false };

  const dirs = entries.map((e) => dirOf(e.trend));
  const allSame = dirs.every((d) => d === dirs[0]);
  const countWord = entries.length === 3 ? 'Les 3 TF' : `Les ${entries.length} TF`;

  if (allSame) {
    if (dirs[0] === 'flat') {
      return { kind: 'neutral', text: `${countWord} sont neutres.`, disagreement: false };
    }
    return {
      kind: 'aligned',
      text: `${countWord} sont alignés (${dirs[0] === 'up' ? 'haussiers' : 'baissiers'}).`,
      disagreement: false,
    };
  }

  // A real « contre » exists only when an up and a down direction coexist.
  const contradiction = dirs.includes('up') && dirs.includes('down');

  // Pullback: the two higher timeframes share one non-flat direction and M15 is
  // the opposite non-flat direction — M15 is pulling back against them.
  const { h4, h1, m15 } = trends;
  if (
    h4 != null &&
    h1 != null &&
    m15 != null &&
    dirOf(h4) === dirOf(h1) &&
    dirOf(h4) !== 'flat' &&
    dirOf(m15) !== 'flat' &&
    dirOf(m15) !== dirOf(h4)
  ) {
    // "tendance" is feminine → "haussière" / "baissière" (not the masculine adj).
    const fem = dirOf(h4) === 'up' ? 'haussière' : 'baissière';
    return {
      kind: 'pullback',
      text: `M15 se replie contre la tendance H4 ${fem}.`,
      disagreement: true,
    };
  }

  // General mix: list each available timeframe's observed trend. It is a
  // disagreement only when a true contradiction (up AND down) is present.
  const parts = entries.map((e) => `${e.label} ${TREND_ADJ[e.trend]}`);
  return {
    kind: contradiction ? 'divergent' : 'partial',
    text: `Les TF divergent : ${joinFr(parts)}.`,
    disagreement: contradiction,
  };
}

/**
 * One present-tense line characterising the RELATION between the timeframes.
 * Thin wrapper over {@link classifyMtfAlignment} — returns just its `text`
 * ('' when no timeframe is available, so the caller hides the line).
 */
export function describeMtfAlignment(trends: MtfTrendMap): string {
  return classifyMtfAlignment(trends).text;
}
