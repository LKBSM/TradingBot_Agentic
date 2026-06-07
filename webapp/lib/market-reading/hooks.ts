'use client';

import * as React from 'react';
import { fetchMarketReading } from './api-client';
import type { MarketReading } from '@/types/market-reading';

export interface UseMarketReadingResult {
  data: MarketReading | null;
  /** True during the initial load for the current combo (no data yet). */
  isLoading: boolean;
  /** True during a background refresh (poll / manual) while stale data shows. */
  isRefreshing: boolean;
  error: Error | null;
  /** Force an out-of-band refresh of the current combo. */
  refresh(): void;
}

export interface UseMarketReadingOptions {
  /** Poll interval in ms. Omit / 0 to disable polling. */
  pollMs?: number;
}

/**
 * Fetch + cache a single market reading for `(instrument, timeframe)`.
 *
 * State management is intentionally light (useState + useEffect, no SWR /
 * React-Query). Behaviour:
 *   · `instrument`/`timeframe` null → idle (no request, no error).
 *   · combo change → blanks data, flips `isLoading` (skeleton).
 *   · poll / manual refresh of the same combo → keeps stale data, flips
 *     `isRefreshing`.
 *   · stale responses (combo changed mid-flight) are discarded.
 */
export function useMarketReading(
  instrument: string | null,
  timeframe: string | null,
  options: UseMarketReadingOptions = {},
): UseMarketReadingResult {
  const { pollMs } = options;

  const [data, setData] = React.useState<MarketReading | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);
  const [isRefreshing, setIsRefreshing] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);

  // Monotonic request token — guards against out-of-order / stale responses.
  const requestSeq = React.useRef(0);
  // Key of the combo whose data is currently held, to tell a combo change
  // (blank + load) from a same-combo refresh (keep + refresh).
  const loadedKey = React.useRef<string | null>(null);
  // Manual-refresh nonce — bumping it re-runs the effect.
  const [refreshNonce, setRefreshNonce] = React.useState(0);

  const refresh = React.useCallback(() => {
    setRefreshNonce((n) => n + 1);
  }, []);

  React.useEffect(() => {
    if (!instrument || !timeframe) {
      // Idle: clear everything and run no request.
      loadedKey.current = null;
      setData(null);
      setIsLoading(false);
      setIsRefreshing(false);
      setError(null);
      return;
    }

    const key = `${instrument}:${timeframe}`;
    const isComboChange = loadedKey.current !== key;
    loadedKey.current = key;

    const seq = ++requestSeq.current;
    const controller = new AbortController();

    setError(null);
    if (isComboChange) {
      setData(null);
      setIsLoading(true);
      setIsRefreshing(false);
    } else {
      setIsRefreshing(true);
    }

    fetchMarketReading(instrument, timeframe, { signal: controller.signal })
      .then((reading) => {
        if (seq !== requestSeq.current) return; // stale
        setData(reading);
        setError(null);
      })
      .catch((err: unknown) => {
        if (seq !== requestSeq.current) return; // stale
        if (controller.signal.aborted) return; // unmounted / superseded
        setError(err instanceof Error ? err : new Error(String(err)));
      })
      .finally(() => {
        if (seq !== requestSeq.current) return; // stale
        setIsLoading(false);
        setIsRefreshing(false);
      });

    return () => controller.abort();
    // refreshNonce is a dependency so refresh() re-triggers the fetch.
  }, [instrument, timeframe, refreshNonce]);

  // Optional polling.
  React.useEffect(() => {
    if (!instrument || !timeframe || !pollMs || pollMs <= 0) return;
    const id = window.setInterval(refresh, pollMs);
    return () => window.clearInterval(id);
  }, [instrument, timeframe, pollMs, refresh]);

  return { data, isLoading, isRefreshing, error, refresh };
}
