'use client';

import * as React from 'react';
import { fetchCandles, fetchMarketReading, MarketReadingNotAvailableError } from './api-client';
import { computeDailyChange, type DailyChange } from './price';
import { getMockCandles, getMockReading, READING_DATA_SOURCE } from '@/lib/mockReadings';
import { MTF_TREND_ORDER, type MtfTrendMap } from './mtf-trend';
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

// ─── Multi-timeframe trend snapshot (read-only) ───────────────────────────────

export interface UseMtfTrendsResult {
  /** M15 / H1 / H4 trend values; each null while unavailable / not loaded. */
  trends: MtfTrendMap;
  isLoading: boolean;
}

const EMPTY_MTF_TRENDS: MtfTrendMap = { h4: null, h1: null, m15: null };

/**
 * Read-only multi-timeframe trend snapshot for `instrument`: the M15 / H1 / H4
 * trend values, each taken from that timeframe's EXISTING market reading
 * (`regime.trend`). It performs NO new detection and NO recompute — it just
 * reads three (cache-served) reads in parallel. A failed / missing timeframe
 * collapses to null so the panel degrades gracefully.
 */
export function useMtfTrends(
  instrument: string | null,
  options: { source?: ReadingSource } = {},
): UseMtfTrendsResult {
  const { source = READING_DATA_SOURCE } = options;
  const [trends, setTrends] = React.useState<MtfTrendMap>(EMPTY_MTF_TRENDS);
  const [isLoading, setIsLoading] = React.useState(false);
  const seqRef = React.useRef(0);

  React.useEffect(() => {
    if (!instrument) {
      setTrends(EMPTY_MTF_TRENDS);
      setIsLoading(false);
      return;
    }

    const seq = ++seqRef.current;
    setIsLoading(true);
    setTrends(EMPTY_MTF_TRENDS);

    // ── Mock source: resolve locally, no network. ──
    if (source === 'mock') {
      const next: MtfTrendMap = { ...EMPTY_MTF_TRENDS };
      for (const { key, tf } of MTF_TREND_ORDER) {
        next[key] = getMockReading(instrument, tf)?.regime.trend ?? null;
      }
      if (seq === seqRef.current) {
        setTrends(next);
        setIsLoading(false);
      }
      return;
    }

    const controller = new AbortController();
    Promise.all(
      MTF_TREND_ORDER.map(({ key, tf }) =>
        fetchMarketReading(instrument, tf, { signal: controller.signal })
          .then((r) => [key, r.regime.trend] as const)
          .catch(() => [key, null] as const),
      ),
    )
      .then((pairs) => {
        if (seq !== seqRef.current) return;
        const next: MtfTrendMap = { ...EMPTY_MTF_TRENDS };
        for (const [key, trend] of pairs) next[key] = trend;
        setTrends(next);
        setIsLoading(false);
      })
      .catch(() => {
        // Defensive: the inner fetches each .catch already, so Promise.all does
        // not reject — but never leave isLoading stuck true if the .then body
        // itself throws (UI-15).
        if (seq === seqRef.current) setIsLoading(false);
      });

    return () => controller.abort();
  }, [instrument, source]);

  return { trends, isLoading };
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
 * Candle depth requested for the chart. A few hundred bars of real history
 * (vs the client default of 200) so the chart shows context, not a keyhole.
 * Served straight from the SQLite candle cache (backend caps at 1000 and the
 * assembler caches 500) — NO extra Twelve Data call. Kept well within the cache
 * so every combo resolves without a provider round-trip.
 */
const CHART_CANDLE_LIMIT = 400;

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

    fetchCandles(instrument, timeframe, {
      signal: controller.signal,
      limit: CHART_CANDLE_LIMIT,
    })
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

// ─── Unified last price (header) ─────────────────────────────────────────────

/**
 * Timeframe the unified header price is read from. M15 is the finest combo
 * served by /api/candles, so its last closed candle is the freshest descriptive
 * price available — identical whatever timeframe the chart shows.
 */
const LATEST_PRICE_TF = 'M15';
/** Window pulled to find the previous-UTC-day reference close (≈ 3 days of M15). */
const LATEST_PRICE_LIMIT = 300;
/**
 * Light refresh cadence for the header price. NOT a tick stream — a coarse
 * cache read (no Twelve Data call) so the header feels alive between candle
 * closes without leaving the "closed-candle" model.
 */
export const DEFAULT_LATEST_PRICE_INTERVAL_MS = 45_000;

export interface UseLatestPriceResult {
  /** Unified last price + descriptive daily change, or null when unavailable. */
  change: DailyChange | null;
  isLoading: boolean;
}

export interface UseLatestPriceOptions {
  source?: ReadingSource;
  /** Active reading's `candle_close_ts` — a fresh close re-pulls the price too. */
  candleCloseTs?: string | null;
  /** Light poll interval in ms (default 45s). Set to 0 to disable polling. */
  intervalMs?: number;
}

/**
 * Resolve ONE unified last price for `instrument`, independent of the displayed
 * timeframe, plus its descriptive daily % change.
 *
 * Always reads the M15 candle window (the freshest closed price) — so the H1/H4
 * header no longer lags behind M15. Pure cache read via /api/candles (no API
 * key, no provider call). Refetches on a light interval AND whenever a candle
 * closes on the active timeframe. In mock mode it derives from the local mock
 * M15 candles; if the feed is unavailable it returns `change: null` and the
 * header falls back to the per-timeframe `close_price`.
 */
export function useLatestPrice(
  instrument: string | null,
  options: UseLatestPriceOptions = {},
): UseLatestPriceResult {
  const {
    source = READING_DATA_SOURCE,
    candleCloseTs = null,
    intervalMs = DEFAULT_LATEST_PRICE_INTERVAL_MS,
  } = options;

  const [change, setChange] = React.useState<DailyChange | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);
  const requestSeq = React.useRef(0);
  const [tick, setTick] = React.useState(0);

  React.useEffect(() => {
    if (!instrument) {
      setChange(null);
      setIsLoading(false);
      return;
    }

    const seq = ++requestSeq.current;

    // ── Mock source: derive from local mock M15 candles, no network. ──
    if (source === 'mock') {
      setChange(computeDailyChange(getMockCandles(instrument, LATEST_PRICE_TF)));
      setIsLoading(false);
      return;
    }

    const controller = new AbortController();
    setIsLoading(true);

    fetchCandles(instrument, LATEST_PRICE_TF, {
      signal: controller.signal,
      limit: LATEST_PRICE_LIMIT,
    })
      .then((data) => {
        if (seq !== requestSeq.current) return; // stale
        setChange(computeDailyChange(data));
      })
      .catch(() => {
        if (seq !== requestSeq.current) return; // stale
        if (controller.signal.aborted) return; // unmounted / superseded
        // Feed unavailable → no unified price → header falls back to close_price.
        setChange(null);
      })
      .finally(() => {
        if (seq !== requestSeq.current) return; // stale
        setIsLoading(false);
      });

    return () => controller.abort();
    // `tick` (interval) and `candleCloseTs` (fresh close) both re-pull the price.
  }, [instrument, source, candleCloseTs, tick]);

  // Light polling — coarse cache read, never a tick stream.
  React.useEffect(() => {
    if (!instrument || source === 'mock' || !intervalMs || intervalMs <= 0) {
      return;
    }
    const id = window.setInterval(() => setTick((t) => t + 1), intervalMs);
    return () => window.clearInterval(id);
  }, [instrument, source, intervalMs]);

  return { change, isLoading };
}
