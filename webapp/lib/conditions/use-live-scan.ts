'use client';

import * as React from 'react';
import type { ConditionsConfig, ConditionsScanResponse } from './types';
import { fetchConditionsScan, ScanNotAvailableError } from './api-client';

/**
 * SC-4 — the FULL scan result, refreshed while the reading is composed.
 *
 * `useLiveComboCount` runs the very same request and then throws all of it away
 * except the count, which was right when results lived behind a second click.
 * SC-4 shows the results as they build, so it needs the whole response — and
 * running both hooks would double every scan. The palette builder keeps
 * `useLiveComboCount`; this is its full-payload sibling, same debounce, same
 * cancellation, same "never fabricate a number" contract.
 *
 * Cheap on purpose: `POST /api/conditions-scan` is a plain SELECT over readings
 * the scheduler already produced (no candles, no detection, no LLM), so the
 * expensive half of the live flow is the translation, never this.
 *
 * DESCRIPTIVE only: it reports what matches, in a fixed order. It never sorts,
 * scores or ranks.
 */

export interface LiveScan {
  status: 'idle' | 'loading' | 'ready' | 'error';
  response: ConditionsScanResponse | null;
  /** Matching combos in the last ready response — `0` is a true reading. */
  count: number;
  scanned: number;
  /** Set when the scan is unavailable rather than merely failing. */
  unavailable: boolean;
}

const IDLE: LiveScan = {
  status: 'idle',
  response: null,
  count: 0,
  scanned: 0,
  unavailable: false,
};

const DEBOUNCE_MS = 400;

export function useLiveScan(config: ConditionsConfig | null): {
  scan: LiveScan;
  refresh(): void;
} {
  const [scan, setScan] = React.useState<LiveScan>(IDLE);
  const [nonce, setNonce] = React.useState(0);
  // Serialised so the effect only re-runs when the composed reading truly changes.
  const key = config && config.conditions.length > 0 ? JSON.stringify(config) : '';

  React.useEffect(() => {
    if (!key || !config || config.conditions.length === 0) {
      setScan(IDLE);
      return;
    }
    let cancelled = false;
    setScan((prev) => ({ ...prev, status: 'loading' }));
    const timer = window.setTimeout(async () => {
      try {
        const response = await fetchConditionsScan(config);
        if (cancelled) return;
        setScan({
          status: 'ready',
          response,
          count: response.matches.filter((m) => m.matched).length,
          scanned: response.scanned,
          unavailable: false,
        });
      } catch (err) {
        if (cancelled) return;
        // Keep the last good results on screen — a transient failure must not
        // blank a reading the user is still composing.
        setScan((prev) => ({
          ...prev,
          status: 'error',
          unavailable: err instanceof ScanNotAvailableError,
        }));
      }
    }, DEBOUNCE_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, nonce]);

  const refresh = React.useCallback(() => setNonce((n) => n + 1), []);
  return { scan, refresh };
}
