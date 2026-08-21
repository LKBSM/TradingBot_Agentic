'use client';

import * as React from 'react';
import { ALL_MARKET_IDS } from '@/lib/markets';

/**
 * Pinned markets — a small client-only preference persisted in localStorage so
 * the user's quick-access markets survive reloads. MKT-1 scope: persistence is
 * LOCAL only (no backend sync — the selector shows a "non synchronisé" mention);
 * the stored value is a list of market ids restricted to the registry perimeter
 * (defensive — never trusts stale storage after a market is removed).
 *
 * This supersedes the per-COMBO pins (mia.pinnedCombos.v1): a favourite is a
 * MARKET, not a market×timeframe pair — the right granularity for a catalogue
 * heading toward 80+ markets.
 */

const STORAGE_KEY = 'mia.pinnedMarkets.v1';

/** All valid market ids — used to sanitise whatever localStorage returns. */
const VALID_IDS = new Set(ALL_MARKET_IDS);

function readStorage(): string[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Keep only known market ids, de-duplicated, in stored order.
    const seen = new Set<string>();
    const out: string[] = [];
    for (const id of parsed) {
      const up = typeof id === 'string' ? id.toUpperCase() : '';
      if (up && VALID_IDS.has(up) && !seen.has(up)) {
        seen.add(up);
        out.push(up);
      }
    }
    return out;
  } catch {
    return [];
  }
}

function writeStorage(ids: string[]): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
  } catch {
    // Quota / privacy mode — pinning degrades to in-memory only.
  }
}

export interface UsePinnedMarketsResult {
  /** Ordered list of pinned market ids (most-recently pinned last). */
  pinned: string[];
  /** Set of pinned market ids, for O(1) membership checks. */
  pinnedSet: ReadonlySet<string>;
  isPinned(id: string): boolean;
  toggle(id: string): void;
}

/**
 * React hook over the pinned-markets store. Hydration-safe: starts empty on the
 * server / first client render, then loads from localStorage in an effect (so
 * SSR markup and the first client paint match). Stays in sync across tabs via
 * the `storage` event.
 */
export function usePinnedMarkets(): UsePinnedMarketsResult {
  const [ids, setIds] = React.useState<string[]>([]);

  React.useEffect(() => {
    setIds(readStorage());
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) setIds(readStorage());
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const toggle = React.useCallback((id: string) => {
    const up = (id ?? '').toUpperCase();
    if (!VALID_IDS.has(up)) return;
    setIds((prev) => {
      const next = prev.includes(up)
        ? prev.filter((k) => k !== up)
        : [...prev, up];
      writeStorage(next);
      return next;
    });
  }, []);

  const pinnedSet = React.useMemo(() => new Set(ids), [ids]);
  const isPinned = React.useCallback((id: string) => pinnedSet.has((id ?? '').toUpperCase()), [pinnedSet]);

  return { pinned: ids, pinnedSet, isPinned, toggle };
}
