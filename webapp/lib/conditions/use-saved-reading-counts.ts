'use client';

import * as React from 'react';
import type { SavedStrategy } from './strategy-store';
import { validateStrategy } from './strategy-store';
import { fetchConditionsScan } from './api-client';

/**
 * Combo counts for saved readings (C4) — re-evaluated ON OPEN, never in the
 * background. Staleness is judged PER COMBO on its OWN timeframe, reusing the
 * scan's ``bars_behind`` (from ``_compute_freshness``): a combo is stale once a
 * candle of ITS unit has closed since the reading was produced. A count is
 * NEVER presented as complete when some of its combos are stale — the caller
 * shows an « Incomplete count » badge with the number of stale combos and a
 * Rescan button. Invalid readings (out-of-schema) are not scanned.
 */

export interface ReadingCount {
  status: 'loading' | 'ready' | 'error' | 'invalid';
  /** Combos that fully match the reading. */
  count: number;
  /** Combos whose reading is behind ≥1 candle of its own unit, plus combos with
   *  no reading — the count is INCOMPLETE by this many. */
  staleCount: number;
  scanned: number;
  /** When this count was evaluated (scan ``as_of``), for « evaluated … ago ». */
  evaluatedAt: string | null;
}

const EMPTY: ReadingCount = {
  status: 'loading', count: 0, staleCount: 0, scanned: 0, evaluatedAt: null,
};

export interface UseSavedReadingCountsResult {
  counts: Record<string, ReadingCount>;
  rescan(id: string): void;
}

export function useSavedReadingCounts(
  strategies: SavedStrategy[],
  active: boolean,
): UseSavedReadingCountsResult {
  const [counts, setCounts] = React.useState<Record<string, ReadingCount>>({});

  const scanOne = React.useCallback(async (s: SavedStrategy) => {
    if (validateStrategy(s).length > 0) {
      setCounts((prev) => ({ ...prev, [s.id]: { ...EMPTY, status: 'invalid' } }));
      return;
    }
    setCounts((prev) => ({ ...prev, [s.id]: { ...(prev[s.id] ?? EMPTY), status: 'loading' } }));
    try {
      const res = await fetchConditionsScan(s.config);
      const count = res.matches.filter((m) => m.matched).length;
      // Per-combo staleness on its own timeframe: bars_behind ≥ 1 means a candle
      // of that unit has closed since the reading. Missing readings count too.
      const staleCount =
        res.matches.filter((m) => (m.bars_behind ?? 0) >= 1).length + res.unavailable.length;
      setCounts((prev) => ({
        ...prev,
        [s.id]: { status: 'ready', count, staleCount, scanned: res.scanned, evaluatedAt: res.as_of },
      }));
    } catch {
      setCounts((prev) => ({ ...prev, [s.id]: { ...(prev[s.id] ?? EMPTY), status: 'error' } }));
    }
  }, []);

  // Re-evaluate ON OPEN (active) and whenever the set of readings changes.
  const ids = strategies.map((s) => s.id).join(',');
  React.useEffect(() => {
    if (!active) return;
    for (const s of strategies) void scanOne(s);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, ids, scanOne]);

  const rescan = React.useCallback(
    (id: string) => {
      const s = strategies.find((x) => x.id === id);
      if (s) void scanOne(s);
    },
    [strategies, scanOne],
  );

  return { counts, rescan };
}
