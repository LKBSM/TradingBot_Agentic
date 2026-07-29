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
import { alignmentTimeframes } from '@/lib/timeframes';

export interface MtfEntry {
  key: string;   // lowercase tf id (map key)
  label: string; // display label (e.g. "H4")
  tf: string;    // uppercase tf id (fetch target)
}

/**
 * The units the MTF alignment tile compares for a GIVEN viewed timeframe (TF-1
 * decision C): the units ABOVE it, highest-relevance first (D1 → W1 at the top;
 * empty when there is no higher unit). No longer a fixed H4·H1·M15 triplet
 * disconnected from what the user is looking at.
 */
export function mtfOrderFor(timeframe: string): MtfEntry[] {
  return alignmentTimeframes(timeframe).map((tf) => ({ key: tf.toLowerCase(), label: tf, tf }));
}

/** Current trend per timeframe; null when that timeframe's read is unavailable. */
export type MtfTrendMap = Record<string, TrendValue | null>;

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
export function classifyMtfAlignment(trends: MtfTrendMap, order: MtfEntry[]): MtfRelation {
  const entries = order
    .map(({ key, label }) => ({ label, trend: trends[key] }))
    .filter((e): e is { label: string; trend: TrendValue } => e.trend != null);

  if (entries.length === 0) return { kind: 'none', text: '', disagreement: false };

  const dirs = entries.map((e) => dirOf(e.trend));
  const allSame = dirs.every((d) => d === dirs[0]);
  const unit = entries.length === 1 ? 'unité supérieure' : 'unités supérieures';
  const countWord = `${entries.length === 1 ? "L'" : 'Les '}${entries.length === 1 ? '' : entries.length + ' '}${unit}`;

  if (allSame) {
    if (dirs[0] === 'flat') {
      const verb = entries.length === 1 ? 'est neutre' : 'sont neutres';
      return { kind: 'neutral', text: `${countWord} ${verb}.`, disagreement: false };
    }
    const adj = dirs[0] === 'up'
      ? entries.length === 1 ? 'haussière' : 'haussières'
      : entries.length === 1 ? 'baissière' : 'baissières';
    const verb = entries.length === 1 ? 'est' : 'sont';
    return { kind: 'aligned', text: `${countWord} ${verb} ${adj}.`, disagreement: false };
  }

  // A real « contre » (disagreement) exists only when an up and a down direction
  // coexist. A direction-vs-flat mix is not a disagreement. Named, descriptive —
  // no wording implies alignment is favourable or better than disagreement.
  const contradiction = dirs.includes('up') && dirs.includes('down');
  const parts = entries.map((e) => `${e.label} ${TREND_ADJ[e.trend]}`);
  return {
    kind: contradiction ? 'divergent' : 'partial',
    text: `Unités supérieures : ${joinFr(parts)}.`,
    disagreement: contradiction,
  };
}

/**
 * One present-tense line characterising the RELATION between the units above the
 * viewed one. Thin wrapper over {@link classifyMtfAlignment} — returns just its
 * `text` ('' when none are available, so the caller hides the line).
 */
export function describeMtfAlignment(trends: MtfTrendMap, order: MtfEntry[]): string {
  return classifyMtfAlignment(trends, order).text;
}
