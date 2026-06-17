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
 * One present-tense line characterising the RELATION between the timeframes,
 * derived purely by classifying the already-computed trend values:
 *   · all same non-flat  → "Les 3 TF sont alignés (haussiers)."
 *   · all flat           → "Les 3 TF sont neutres."
 *   · H4+H1 agree, M15 opposes → "M15 se replie contre la tendance H4 haussière."
 *   · otherwise          → "Les TF divergent : H4 haussier, H1 neutre et M15 baissier."
 * Returns '' when no timeframe is available (the caller hides the line). Strictly
 * descriptive — no future tense, no probability, no score, no action verdict.
 */
export function describeMtfAlignment(trends: MtfTrendMap): string {
  const entries = MTF_TREND_ORDER.map(({ key, label }) => ({
    label: label as string,
    trend: trends[key],
  })).filter(
    (e): e is { label: string; trend: TrendValue } => e.trend != null,
  );

  if (entries.length === 0) return '';

  const dirs = entries.map((e) => dirOf(e.trend));
  const allSame = dirs.every((d) => d === dirs[0]);
  const countWord = entries.length === 3 ? 'Les 3 TF' : `Les ${entries.length} TF`;

  if (allSame) {
    if (dirs[0] === 'flat') return `${countWord} sont neutres.`;
    return `${countWord} sont alignés (${dirs[0] === 'up' ? 'haussiers' : 'baissiers'}).`;
  }

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
    return `M15 se replie contre la tendance H4 ${fem}.`;
  }

  // General divergence: list each available timeframe's observed trend.
  const parts = entries.map((e) => `${e.label} ${TREND_ADJ[e.trend]}`);
  return `Les TF divergent : ${joinFr(parts)}.`;
}
