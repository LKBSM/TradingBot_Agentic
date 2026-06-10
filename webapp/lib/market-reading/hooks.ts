'use client';

import * as React from 'react';
import { fetchCandles, fetchMarketReading, MarketReadingNotAvailableError } from './api-client';
import { getMockCandles, getMockReading, READING_DATA_SOURCE } from '@/lib/mockReadings';
import type { Candle, MarketReading } from '@/types/market-reading';

/** Where the reading comes from: live backend or the local TEMPORARY mocks. */
export type ReadingSource = 'live' | 'mock';

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
  /**
   * Data source. Defaults to the module-level READING_DATA_SOURCE flag.
   *   · 'live' → fetchMarketReading() (real backend).
   *   · 'mock' → local TEMPORARY mocks (getMockReading); no network call.
   * Passing it explicitly is mostly useful in tests.
   */
  source?: ReadingSource;
}

/** Simulated latency for the mock source so the skeleton is briefly visible. */
const MOCK_LATENCY_MS = 220;

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
 *
 * The `source` option swaps the backend for the TEMPORARY local mocks (the
 * single swap point for the "produit fini" demo; see lib/mockReadings.ts).
 */
export function useMarketReading(
  instrument: string | null,
  timeframe: string | null,
  options: UseMarketReadingOptions = {},
): UseMarketReadingResult {
  const { pollMs, source = READING_DATA_SOURCE } = options;

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

    // ── Mock source: resolve locally, no network. TEMPORAIRE (cf. mockReadings). ──
    if (source === 'mock') {
      const timer = setTimeout(() => {
        if (seq !== requestSeq.current) return; // stale
        const mock = getMockReading(instrument, timeframe);
        if (mock) {
          setData(mock);
          setError(null);
        } else {
          // No mock for this combo → surface the "unavailable" placeholder.
          setError(
            new MarketReadingNotAvailableError(
              'Lecture indisponible pour cette combinaison.',
            ),
          );
        }
        setIsLoading(false);
        setIsRefreshing(false);
      }, MOCK_LATENCY_MS);
      return () => clearTimeout(timer);
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
  }, [instrument, timeframe, refreshNonce, source]);

  // Optional polling.
  React.useEffect(() => {
    if (!instrument || !timeframe || !pollMs || pollMs <= 0) return;
    const id = window.setInterval(refresh, pollMs);
    return () => window.clearInterval(id);
  }, [instrument, timeframe, pollMs, refresh]);

  return { data, isLoading, isRefreshing, error, refresh };
}

// ─── Candles (chart feed) ─────────────────────────────────────────────────────

export interface UseCandlesResult {
  /** Ascending OHLC window, or null when the feed is unavailable / not loaded. */
  candles: Candle[] | null;
  isLoading: boolean;
  /** Set when the live feed errored (404/400/503/transport). null in mock mode. */
  error: Error | null;
}

export interface UseCandlesOptions {
  /** Data source. Defaults to the module-level READING_DATA_SOURCE flag. */
  source?: ReadingSource;
  /**
   * The active reading's `candle_close_ts`. In live mode the feed is re-fetched
   * only when this changes (or the combo changes), so the chart never polls the
   * backend faster than candles actually close — cheap SQLite read, no Twelve
   * Data call, but it keeps the refresh honest with the "last closed candle".
   */
  candleCloseTs?: string | null;
}

/**
 * Fetch the candle window for `(instrument, timeframe)` for the chart.
 *
 *   · 'mock' → getMockCandles() (deterministic local series; no network).
 *   · 'live' → fetchCandles() (GET /api/candles, descriptive OHLC only).
 *
 * Any live failure (no cache yet, out of perimeter, store down, transport)
 * collapses to `candles: null`, which the column renders as the
 * "graphique indisponible" placeholder — the textual reading stays usable.
 */
export function useCandles(
  instrument: string | null,
  timeframe: string | null,
  options: UseCandlesOptions = {},
): UseCandlesResult {
  const { source = READING_DATA_SOURCE, candleCloseTs = null } = options;

  const [candles, setCandles] = React.useState<Candle[] | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);
  const requestSeq = React.useRef(0);

  React.useEffect(() => {
    if (!instrument || !timeframe) {
      setCandles(null);
      setIsLoading(false);
      setError(null);
      return;
    }

    const seq = ++requestSeq.current;

    // ── Mock source: resolve locally, no network. TEMPORAIRE (cf. mockReadings). ──
    if (source === 'mock') {
      setCandles(getMockCandles(instrument, timeframe));
      setIsLoading(false);
      setError(null);
      return;
    }

    const controller = new AbortController();
    setIsLoading(true);
    setError(null);

    fetchCandles(instrument, timeframe, { signal: controller.signal })
      .then((data) => {
        if (seq !== requestSeq.current) return; // stale
        setCandles(data.length > 0 ? data : null);
        setError(null);
      })
      .catch((err: unknown) => {
        if (seq !== requestSeq.current) return; // stale
        if (controller.signal.aborted) return; // unmounted / superseded
        // Unavailable feed → no candles → placeholder. Keep the error for callers.
        setCandles(null);
        setError(err instanceof Error ? err : new Error(String(err)));
      })
      .finally(() => {
        if (seq !== requestSeq.current) return; // stale
        setIsLoading(false);
      });

    return () => controller.abort();
    // candleCloseTs is a dependency so a freshly-closed candle re-pulls the feed.
  }, [instrument, timeframe, source, candleCloseTs]);

  return { candles, isLoading, error };
}
