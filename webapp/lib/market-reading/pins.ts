'use client';

import * as React from 'react';
import {
  comboKey,
  SUPPORTED_COMBOS,
  type Combo,
} from '@/lib/market-reading/store';

/**
 * Pinned combos — small client-only preference persisted in localStorage so the
 * user's quick-access markets survive reloads. V1 scope: persistence is local
 * only (no backend sync); the stored value is a list of combo keys restricted
 * to the supported V1 perimeter (defensive — never trusts stale storage).
 */

const STORAGE_KEY = 'mia.pinnedCombos.v1';

/** All valid combo keys — used to sanitise whatever localStorage returns. */
const VALID_KEYS = new Set(SUPPORTED_COMBOS.map(comboKey));

function readStorage(): string[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Keep only known combo keys, de-duplicated, in stored order.
    const seen = new Set<string>();
    const out: string[] = [];
    for (const k of parsed) {
      if (typeof k === 'string' && VALID_KEYS.has(k) && !seen.has(k)) {
        seen.add(k);
        out.push(k);
      }
    }
    return out;
  } catch {
    return [];
  }
}

function writeStorage(keys: string[]): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(keys));
  } catch {
    // Quota / privacy mode — pinning degrades to in-memory only.
  }
}

export interface UsePinnedCombosResult {
  /** Ordered list of pinned combos (most-recently pinned last). */
  pinned: Combo[];
  /** Set of pinned combo keys, for O(1) membership checks. */
  pinnedKeys: ReadonlySet<string>;
  isPinned(combo: Combo): boolean;
  toggle(combo: Combo): void;
}

/**
 * React hook over the pinned-combos store. Hydration-safe: starts empty on the
 * server / first client render, then loads from localStorage in an effect (so
 * SSR markup and the first client paint match). Stays in sync across tabs via
 * the `storage` event.
 */
export function usePinnedCombos(): UsePinnedCombosResult {
  const [keys, setKeys] = React.useState<string[]>([]);

  // Load once on mount (post-hydration) and subscribe to cross-tab changes.
  React.useEffect(() => {
    setKeys(readStorage());
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) setKeys(readStorage());
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const toggle = React.useCallback((combo: Combo) => {
    const key = comboKey(combo);
    setKeys((prev) => {
      const next = prev.includes(key)
        ? prev.filter((k) => k !== key)
        : [...prev, key];
      writeStorage(next);
      return next;
    });
  }, []);

  const pinnedKeys = React.useMemo(() => new Set(keys), [keys]);

  const pinned = React.useMemo(
    () =>
      keys
        .map((k) => SUPPORTED_COMBOS.find((c) => comboKey(c) === k))
        .filter((c): c is Combo => c !== undefined),
    [keys],
  );

  const isPinned = React.useCallback(
    (combo: Combo) => pinnedKeys.has(comboKey(combo)),
    [pinnedKeys],
  );

  return { pinned, pinnedKeys, isPinned, toggle };
}
