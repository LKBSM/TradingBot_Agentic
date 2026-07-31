'use client';

import * as React from 'react';
import type { ConditionsConfig } from './types';
import { fetchConditionsScan } from './api-client';

/**
 * Live count of matching combos WHILE the user composes their reading — what
 * makes the builder feel alive as conditions are ticked. Debounced and
 * cancellable so a burst of clicks triggers a single read-only scan. Returns
 * ``idle`` (never a fabricated count) until there is at least one condition.
 * This is DESCRIPTIVE only — it counts, it never ranks or sorts.
 */

export interface LiveComboCount {
  status: 'idle' | 'loading' | 'ready' | 'error';
  count: number;
  scanned: number;
  asOf: string | null;
}

const IDLE: LiveComboCount = { status: 'idle', count: 0, scanned: 0, asOf: null };
const DEBOUNCE_MS = 400;

export function useLiveComboCount(config: ConditionsConfig | null): {
  live: LiveComboCount;
  refresh(): void;
} {
  const [live, setLive] = React.useState<LiveComboCount>(IDLE);
  const [nonce, setNonce] = React.useState(0);
  // Serialise so the effect only re-runs when the composed reading truly changes.
  const key = config && config.conditions.length > 0 ? JSON.stringify(config) : '';

  React.useEffect(() => {
    if (!config || config.conditions.length === 0) {
      setLive(IDLE);
      return;
    }
    let cancelled = false;
    setLive((prev) => ({ ...prev, status: 'loading' }));
    const timer = setTimeout(async () => {
      try {
        const res = await fetchConditionsScan(config);
        if (cancelled) return;
        setLive({
          status: 'ready',
          count: res.matches.filter((m) => m.matched).length,
          scanned: res.scanned,
          asOf: res.as_of,
        });
      } catch {
        if (!cancelled) setLive((prev) => ({ ...prev, status: 'error' }));
      }
    }, DEBOUNCE_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, nonce]);

  const refresh = React.useCallback(() => setNonce((n) => n + 1), []);
  return { live, refresh };
}
