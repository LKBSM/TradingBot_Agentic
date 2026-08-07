'use client';

import * as React from 'react';
import {
  CandlesError,
  fetchCandles,
  fetchCandlesPage,
  fetchMarketReading,
  MarketReadingNotAvailableError,
  type CandlesErrorReason,
} from './api-client';
import { computeDailyChange, type DailyChange } from './price';
import { getMockCandles, getMockReading, READING_DATA_SOURCE } from '@/lib/mockReadings';
import { mtfOrderFor, type MtfTrendMap } from './mtf-trend';
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
 * PERF-1 — in-memory retention across combo switches AND navigation
 * (unmount/remount when leaving /app for Scanner/Zones and back). Keyed by
 * source+instrument+timeframe. On a revisit the last value for that exact combo
 * shows INSTANTLY while a background revalidation refreshes it (SWR-style): the
 * `isRefreshing` indicator is the honesty signal, and every cache hit is
 * revalidated, so nothing stale is ever shown WITHOUT saying so. Bounded by the
 * combo count (≤ 6 instruments × 6 timeframes), so no eviction is needed; it is
 * process-memory only (never persisted) and dies with the tab.
 */
const readingCache = new Map<string, MarketReading>();
const candlesCache = new Map<string, Candle[]>();
// CHART-1: per-combo "older candles still exist before the oldest loaded bar".
// Cached alongside the series so a combo revisit restores the pagination floor
// (and the "start of available data" message) without a re-probe.
const candlesHasMoreCache = new Map<string, boolean>();
const comboCacheKey = (source: string, instrument: string, timeframe: string) =>
  `${source}:${instrument}:${timeframe}`;

/**
 * Test-only: clear the retention caches so a test's initial-load assertions
 * aren't seeded by a previous test in the same file (the caches are module
 * state, shared across tests within a file). Call it in `beforeEach`.
 */
export function __resetReadingRetention(): void {
  readingCache.clear();
  candlesCache.clear();
  candlesHasMoreCache.clear();
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
    const ck = comboCacheKey(source, instrument, timeframe);

    setError(null);
    if (isComboChange) {
      // PERF-1: on a revisit (TF/instrument switch, or return from another page)
      // seed the last known reading for THIS combo instantly and revalidate in
      // the background — no blank skeleton, no full re-wait. A first visit still
      // blanks + shows the skeleton.
      const cached = readingCache.get(ck) ?? null;
      if (cached) {
        setData(cached);
        setIsLoading(false);
        setIsRefreshing(true);
      } else {
        setData(null);
        setIsLoading(true);
        setIsRefreshing(false);
      }
    } else {
      setIsRefreshing(true);
    }

    // ── Mock source: resolve locally, no network. TEMPORAIRE (cf. mockReadings). ──
    if (source === 'mock') {
      const timer = setTimeout(() => {
        if (seq !== requestSeq.current) return; // stale
        const mock = getMockReading(instrument, timeframe);
        if (mock) {
          readingCache.set(ck, mock);
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
        readingCache.set(ck, reading);
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

const EMPTY_MTF_TRENDS: MtfTrendMap = {};

/**
 * Read-only multi-timeframe trend snapshot for `instrument`, RELATIVE to the
 * viewed `timeframe` (TF-1 decision C): the trend of each unit ABOVE it, taken
 * from that unit's EXISTING market reading (`regime.trend`). NO new detection, no
 * recompute — just the (cache-served) upper reads in parallel. A failed/missing
 * unit collapses to null (counted indisponible, never an agreement). At the top
 * of the ladder there is no higher unit and the set is empty.
 */
export function useMtfTrends(
  instrument: string | null,
  timeframe: string | null,
  options: { source?: ReadingSource } = {},
): UseMtfTrendsResult {
  const { source = READING_DATA_SOURCE } = options;
  const [trends, setTrends] = React.useState<MtfTrendMap>(EMPTY_MTF_TRENDS);
  const [isLoading, setIsLoading] = React.useState(false);
  const seqRef = React.useRef(0);

  React.useEffect(() => {
    if (!instrument || !timeframe) {
      setTrends(EMPTY_MTF_TRENDS);
      setIsLoading(false);
      return;
    }

    const order = mtfOrderFor(timeframe);
    const seq = ++seqRef.current;
    setIsLoading(true);
    setTrends(EMPTY_MTF_TRENDS);

    // ── Mock source: resolve locally, no network. ──
    if (source === 'mock') {
      const next: MtfTrendMap = {};
      for (const { key, tf } of order) {
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
      order.map(({ key, tf }) =>
        fetchMarketReading(instrument, tf, { signal: controller.signal })
          .then((r) => [key, r.regime.trend] as const)
          .catch(() => [key, null] as const),
      ),
    )
      .then((pairs) => {
        if (seq !== seqRef.current) return;
        const next: MtfTrendMap = {};
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
  }, [instrument, timeframe, source]);

  return { trends, isLoading };
}

// ─── Candles (chart feed) ─────────────────────────────────────────────────────

export interface UseCandlesResult {
  /** Ascending OHLC window, or null when the feed is unavailable / not loaded. */
  candles: Candle[] | null;
  isLoading: boolean;
  /** Set when the live feed errored (404/400/503/transport). null in mock mode. */
  error: Error | null;
  /**
   * Force an out-of-band re-pull of the candle window WITHOUT reloading the page
   * (PERF-2). The chart's own recovery handle — the reading's `candleCloseTs` only
   * advances at a candle close (≤15 min on M15, up to a day on D1), so a transient
   * candle-feed failure used to leave the chart blank until a manual browser
   * refresh. `refresh()` (and the bounded auto-retry below) close that gap.
   */
  refresh(): void;
  // ── CHART-1 history pagination ──────────────────────────────────────────────
  /** True while older candles still exist before the oldest loaded bar. */
  hasMoreHistory: boolean;
  /** True while a history page is being fetched (older bars). */
  loadingOlder: boolean;
  /** The last history-load error (for a discreet "réessayer"); null otherwise. */
  olderError: Error | null;
  /**
   * Load the page of candles just OLDER than the current oldest, merging it in
   * WITHOUT ever dropping the bars already shown (CHART-1). No-op while a page is
   * in flight, when there is nothing older, or before any candles are loaded.
   */
  loadOlder(): void;
}

/**
 * Merge two ascending candle arrays by `time` (union, `overlay` wins on a tie so
 * a refreshed recent bar overwrites a stale one), returning ascending order.
 * Used both to prepend history and to fold a fresh window into loaded history
 * without losing what the user scrolled back to.
 */
function mergeCandlesByTime(base: Candle[], overlay: Candle[]): Candle[] {
  const byTime = new Map<number, Candle>();
  for (const c of base) byTime.set(c.time, c);
  for (const c of overlay) byTime.set(c.time, c);
  return Array.from(byTime.values()).sort((a, b) => a.time - b.time);
}

/**
 * PERF-2 — bounded auto-retry for the candle feed. A transient transport failure
 * (timeout / network / 5xx) schedules up to this many silent re-pulls with linear
 * backoff, so the chart heals itself without waiting for the next candle close or
 * a manual page reload. A deterministic 404 ("no candles for this combo") is NOT
 * retried automatically — it is surfaced honestly with a manual "Réessayer".
 */
const CANDLES_MAX_AUTO_RETRIES = 3;
const CANDLES_RETRY_BASE_MS = 1_500;
const candlesReasonOf = (err: unknown): CandlesErrorReason | null =>
  err instanceof CandlesError ? err.reason : null;
const isTransientCandlesError = (err: unknown): boolean => {
  const reason = candlesReasonOf(err);
  return reason === 'timeout' || reason === 'network' || reason === 'server';
};

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
  const loadedKey = React.useRef<string | null>(null);

  // ── CHART-1 history pagination state ────────────────────────────────────────
  const [hasMoreHistory, setHasMoreHistory] = React.useState(false);
  const [loadingOlder, setLoadingOlder] = React.useState(false);
  const [olderError, setOlderError] = React.useState<Error | null>(null);
  // Refs so the stable `loadOlder` callback always sees the latest values
  // without re-creating itself (the chart wires it to a zoom-out handler).
  const candlesRef = React.useRef<Candle[] | null>(null);
  const hasMoreRef = React.useRef(false);
  const loadingOlderRef = React.useRef(false);
  const instrumentRef = React.useRef(instrument);
  const timeframeRef = React.useRef(timeframe);
  const olderAbortRef = React.useRef<AbortController | null>(null);
  React.useEffect(() => {
    candlesRef.current = candles;
  }, [candles]);
  React.useEffect(() => {
    hasMoreRef.current = hasMoreHistory;
  }, [hasMoreHistory]);
  React.useEffect(() => {
    loadingOlderRef.current = loadingOlder;
  }, [loadingOlder]);
  instrumentRef.current = instrument;
  timeframeRef.current = timeframe;

  // PERF-2 recovery handles: a manual-refresh nonce (bumped by refresh() and by
  // the bounded auto-retry) re-runs the effect; the attempt counter is reset on
  // every combo change and every success so each combo gets its own retry budget.
  const [refreshNonce, setRefreshNonce] = React.useState(0);
  const retryCountRef = React.useRef(0);
  const retryTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const refresh = React.useCallback(() => {
    retryCountRef.current = 0; // a manual retry restores the full auto-retry budget
    setRefreshNonce((n) => n + 1);
  }, []);

  React.useEffect(() => {
    if (!instrument || !timeframe) {
      loadedKey.current = null;
      setCandles(null);
      setIsLoading(false);
      setError(null);
      setHasMoreHistory(false);
      setLoadingOlder(false);
      setOlderError(null);
      return;
    }

    const seq = ++requestSeq.current;
    const key = `${instrument}:${timeframe}`;
    const isComboChange = loadedKey.current !== key;
    loadedKey.current = key;
    const ck = comboCacheKey(source, instrument, timeframe);
    // A combo change starts a fresh retry budget (a new series, new failures).
    if (isComboChange) retryCountRef.current = 0;

    // ── Mock source: resolve locally, no network. TEMPORAIRE (cf. mockReadings). ──
    if (source === 'mock') {
      setCandles(getMockCandles(instrument, timeframe));
      setIsLoading(false);
      setError(null);
      setHasMoreHistory(false);
      return;
    }

    // PERF-1: on a combo revisit, show THIS combo's cached candles instantly
    // (correct series, no wrong-combo flash) and revalidate below. A same-combo
    // re-pull (a candle just closed) keeps the current series while it refreshes.
    // CHART-1: a combo change abandons any in-flight history load and restores
    // THIS combo's pagination floor from cache.
    if (isComboChange) {
      olderAbortRef.current?.abort();
      setLoadingOlder(false);
      setOlderError(null);
      const cached = candlesCache.get(ck) ?? null;
      if (cached) setCandles(cached);
      setHasMoreHistory(candlesHasMoreCache.get(ck) ?? false);
    }

    const controller = new AbortController();
    setIsLoading(true);
    setError(null);

    fetchCandlesPage(instrument, timeframe, {
      signal: controller.signal,
      limit: CHART_CANDLE_LIMIT,
    })
      .then(({ candles: data, hasMore }) => {
        if (seq !== requestSeq.current) return; // stale
        retryCountRef.current = 0; // healed — restore the budget for future blips
        if (data.length === 0) {
          setCandles(null);
          setError(null);
          return;
        }
        // CHART-1: fold the fresh latest window into whatever is already loaded
        // (which may include history the user scrolled back to) rather than
        // replacing it — loaded bars must never vanish on a candle close. Only a
        // brand-new combo with no loaded/cached series takes the window verbatim.
        const existing = isComboChange ? candlesCache.get(ck) ?? null : candlesRef.current;
        const merged = existing && existing.length > 0 ? mergeCandlesByTime(existing, data) : data;
        candlesCache.set(ck, merged);
        setCandles(merged);
        // `hasMore` from this window is about older-than-the-window; it only
        // reflects our true floor when nothing older is already loaded. If we
        // already hold history older than the window, keep the known floor.
        const oldestLoaded = merged[0]?.time ?? null;
        const windowOldest = data[0]?.time ?? null;
        if (oldestLoaded != null && windowOldest != null && oldestLoaded < windowOldest) {
          // history already extends past the window — floor unchanged.
          setHasMoreHistory(candlesHasMoreCache.get(ck) ?? hasMoreRef.current);
        } else {
          candlesHasMoreCache.set(ck, hasMore);
          setHasMoreHistory(hasMore);
        }
        setError(null);
      })
      .catch((err: unknown) => {
        if (seq !== requestSeq.current) return; // stale
        if (controller.signal.aborted) return; // unmounted / superseded
        // Unavailable feed → no candles → placeholder. Keep the error for callers.
        setCandles(null);
        setError(err instanceof Error ? err : new Error(String(err)));
        // PERF-2: heal a TRANSIENT failure without waiting for the next candle
        // close (candleCloseTs) or a page reload. Bounded, linear backoff. A
        // deterministic 404 is left for the manual "Réessayer" (retry is futile
        // until the backend actually has candles).
        if (
          isTransientCandlesError(err) &&
          retryCountRef.current < CANDLES_MAX_AUTO_RETRIES
        ) {
          retryCountRef.current += 1;
          const delay = CANDLES_RETRY_BASE_MS * retryCountRef.current;
          if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
          retryTimerRef.current = setTimeout(() => {
            if (seq !== requestSeq.current) return; // combo moved on
            setRefreshNonce((n) => n + 1);
          }, delay);
        }
      })
      .finally(() => {
        if (seq !== requestSeq.current) return; // stale
        setIsLoading(false);
      });

    return () => {
      controller.abort();
      if (retryTimerRef.current) {
        clearTimeout(retryTimerRef.current);
        retryTimerRef.current = null;
      }
    };
    // candleCloseTs re-pulls on a fresh close; refreshNonce on manual/auto retry.
  }, [instrument, timeframe, source, candleCloseTs, refreshNonce]);

  // ── CHART-1: load the page just OLDER than the current oldest, on demand ──────
  const loadOlder = React.useCallback(() => {
    if (source === 'mock') return;
    const inst = instrumentRef.current;
    const tf = timeframeRef.current;
    const loaded = candlesRef.current;
    if (!inst || !tf || !loaded || loaded.length === 0) return;
    if (loadingOlderRef.current || !hasMoreRef.current) return;

    const oldest = loaded[0]!.time;
    const ck = comboCacheKey(source, inst, tf);
    const controller = new AbortController();
    olderAbortRef.current?.abort();
    olderAbortRef.current = controller;
    setLoadingOlder(true);
    setOlderError(null);

    fetchCandlesPage(inst, tf, {
      signal: controller.signal,
      limit: CHART_CANDLE_LIMIT,
      before: oldest,
    })
      .then(({ candles: older, hasMore }) => {
        if (controller.signal.aborted) return;
        // Guard against a combo change mid-flight.
        if (instrumentRef.current !== inst || timeframeRef.current !== tf) return;
        const current = candlesRef.current ?? [];
        // Prepend the older page; merge preserves everything already shown.
        const merged = older.length > 0 ? mergeCandlesByTime(older, current) : current;
        candlesCache.set(ck, merged);
        candlesHasMoreCache.set(ck, hasMore);
        setCandles(merged);
        setHasMoreHistory(hasMore);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        if (instrumentRef.current !== inst || timeframeRef.current !== tf) return;
        // The already-shown candles stay; surface a discreet retry affordance.
        setOlderError(err instanceof Error ? err : new Error(String(err)));
      })
      .finally(() => {
        if (controller.signal.aborted) return;
        setLoadingOlder(false);
      });
  }, [source]);

  return {
    candles,
    isLoading,
    error,
    refresh,
    hasMoreHistory,
    loadingOlder,
    olderError,
    loadOlder,
  };
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
  // Freshness floor (UI-05): the seq guard orders by request emission, but a
  // newer request can still return an OLDER candle from the cache. Never let the
  // displayed price rewind to an earlier timestamp. Reset on instrument change.
  const lastPriceTsRef = React.useRef(0);
  React.useEffect(() => {
    lastPriceTsRef.current = 0;
  }, [instrument]);

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
        const next = computeDailyChange(data);
        // Drop a result that would rewind the price to an older candle (UI-05).
        if (next && next.priceTs < lastPriceTsRef.current) return;
        if (next) lastPriceTsRef.current = next.priceTs;
        setChange(next);
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
