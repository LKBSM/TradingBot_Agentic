import sampleSignalsJson from '@/mocks/sample_signals.json';
import type { InsightSignalV2 } from '@/types/insight';

/**
 * Typed view of the static demo signals shipped with the webapp. These are
 * NOT live data — they exist so the UI components can be exercised before
 * the backend integration sprint. Index 0 = bullish XAU M15 hero example,
 * 1 = bearish EURUSD H1 (ranging regime, jump-heavy), 2 = neutral XAU H4
 * (consolidation).
 */
// JSON-imported tuples widen to `number[]`; the demo file is hand-curated to
// match InsightSignalV2 exactly, so an unsafe assertion is acceptable here.
// When the backend integration sprint lands, validate with zod at the boundary.
export const SAMPLE_SIGNALS = sampleSignalsJson as unknown as readonly InsightSignalV2[];

export function getSampleSignalById(id: string): InsightSignalV2 | undefined {
  return SAMPLE_SIGNALS.find((s) => s.id === id);
}

export function getHeroSampleSignal(): InsightSignalV2 {
  const first = SAMPLE_SIGNALS[0];
  if (!first) {
    throw new Error('mocks/sample_signals.json must contain at least one entry');
  }
  return first;
}
