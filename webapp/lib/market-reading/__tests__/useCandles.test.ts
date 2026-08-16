import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Network + mock-candle layers are stubbed so the hook is tested in isolation.
// Keep the real error classes (CandlesError) — the hook references them at module
// scope (isTransientCandlesError) and for `instanceof` reason checks. The chart
// feed goes through fetchCandleWindow (candles + backward-paging flag).
vi.mock('../api-client', async (importActual) => {
  const actual = await importActual<typeof import('../api-client')>();
  return { ...actual, fetchCandleWindow: vi.fn() };
});
vi.mock('@/lib/mockReadings', () => ({
  READING_DATA_SOURCE: 'live',
  getMockCandles: vi.fn(),
}));

import { useCandles, __resetReadingRetention } from '../hooks';
import { CandlesError, fetchCandleWindow } from '../api-client';
import { getMockCandles } from '@/lib/mockReadings';
import type { Candle } from '@/types/market-reading';

const mockFetchWindow = vi.mocked(fetchCandleWindow);
const mockGetMockCandles = vi.mocked(getMockCandles);

const c = (time: number): Candle => ({ time, open: 1, high: 2, low: 0.5, close: 1.5 });
const SERIES: Candle[] = [c(1), c(2)];
/** Wrap a candle list into the window envelope the hook now consumes. */
const win = (candles: Candle[], hasMoreHistory = false) => ({ candles, hasMoreHistory });

beforeEach(() => {
  mockFetchWindow.mockReset();
  mockGetMockCandles.mockReset();
  // Module-state retention caches — clear so each test starts cold.
  __resetReadingRetention();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('useCandles', () => {
  it('stays idle (no fetch) when the combo is null', () => {
    const { result } = renderHook(() => useCandles(null, null, { source: 'live' }));
    expect(result.current.candles).toBeNull();
    expect(mockFetchWindow).not.toHaveBeenCalled();
  });

  it('resolves locally in mock mode without any network call', () => {
    mockGetMockCandles.mockReturnValue(SERIES);
    const { result } = renderHook(() =>
      useCandles('XAUUSD', 'M15', { source: 'mock' }),
    );
    expect(result.current.candles).toEqual(SERIES);
    // A mock series is complete — nothing to page back into.
    expect(result.current.reachedStart).toBe(true);
    expect(mockFetchWindow).not.toHaveBeenCalled();
    expect(mockGetMockCandles).toHaveBeenCalledWith('XAUUSD', 'M15');
  });

  it('fetches the live feed and exposes the candles', async () => {
    mockFetchWindow.mockResolvedValue(win(SERIES, true));
    const { result } = renderHook(() =>
      useCandles('XAUUSD', 'M15', { source: 'live' }),
    );
    await waitFor(() => expect(result.current.candles).toEqual(SERIES));
    expect(result.current.error).toBeNull();
    expect(mockFetchWindow).toHaveBeenCalledWith(
      'XAUUSD',
      'M15',
      // Requests a few-hundred-bar window (cache-served, no extra provider call).
      expect.objectContaining({ signal: expect.any(AbortSignal), limit: 400 }),
    );
  });

  it('collapses an unavailable feed to null candles + an error', async () => {
    mockFetchWindow.mockRejectedValue(new Error('no candles cached'));
    const { result } = renderHook(() =>
      useCandles('EURUSD', 'H4', { source: 'live' }),
    );
    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.candles).toBeNull();
  });

  it('treats an empty live window as unavailable (null)', async () => {
    mockFetchWindow.mockResolvedValue(win([]));
    const { result } = renderHook(() =>
      useCandles('XAUUSD', 'H1', { source: 'live' }),
    );
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.candles).toBeNull();
  });

  it('re-fetches when candle_close_ts changes', async () => {
    mockFetchWindow.mockResolvedValue(win(SERIES));
    const { rerender } = renderHook(
      ({ ts }) => useCandles('XAUUSD', 'M15', { source: 'live', candleCloseTs: ts }),
      { initialProps: { ts: '2026-05-26T11:00:00+00:00' } },
    );
    await waitFor(() => expect(mockFetchWindow).toHaveBeenCalledTimes(1));
    rerender({ ts: '2026-05-26T11:15:00+00:00' });
    await waitFor(() => expect(mockFetchWindow).toHaveBeenCalledTimes(2));
  });

  // ── CHART-2: on-demand backward paging ────────────────────────────────────────
  describe('CHART-2 — backward history paging (loadOlder)', () => {
    it('marks reachedStart immediately when the first window is already complete', async () => {
      mockFetchWindow.mockResolvedValue(win(SERIES, false));
      const { result } = renderHook(() =>
        useCandles('XAUUSD', 'M15', { source: 'live' }),
      );
      await waitFor(() => expect(result.current.candles).toEqual(SERIES));
      expect(result.current.reachedStart).toBe(true);
    });

    it('prepends an older page WITHOUT dropping the visible candles, and stops at the real start', async () => {
      const recent = [c(100), c(101)];
      const older = [c(98), c(99)];
      mockFetchWindow
        .mockResolvedValueOnce(win(recent, true)) // initial: more history exists
        .mockResolvedValueOnce(win(older, false)); // older page: reaches the start
      const { result } = renderHook(() =>
        useCandles('XAUUSD', 'M15', { source: 'live' }),
      );
      await waitFor(() => expect(result.current.candles).toEqual(recent));
      expect(result.current.reachedStart).toBe(false);

      act(() => {
        result.current.loadOlder();
      });
      // The already-loaded candles never disappear — the older page is prepended.
      await waitFor(() => expect(result.current.candles).toEqual([...older, ...recent]));
      expect(result.current.reachedStart).toBe(true);
      // Paged from the oldest loaded bar, one dedicated page size.
      expect(mockFetchWindow).toHaveBeenNthCalledWith(
        2,
        'XAUUSD',
        'M15',
        expect.objectContaining({ before: 100, limit: 500 }),
      );

      // Once the start is reached, further loadOlder() calls are no-ops.
      act(() => {
        result.current.loadOlder();
      });
      expect(mockFetchWindow).toHaveBeenCalledTimes(2);
    });

    it('surfaces an older-page failure and keeps the existing candles', async () => {
      const recent = [c(100), c(101)];
      mockFetchWindow
        .mockResolvedValueOnce(win(recent, true))
        .mockRejectedValueOnce(new CandlesError(0, 'timed out', 'timeout'));
      const { result } = renderHook(() =>
        useCandles('XAUUSD', 'M15', { source: 'live' }),
      );
      await waitFor(() => expect(result.current.candles).toEqual(recent));
      act(() => {
        result.current.loadOlder();
      });
      await waitFor(() => expect(result.current.olderError).not.toBeNull());
      // The visible candles are untouched by a failed backward page.
      expect(result.current.candles).toEqual(recent);
      expect(result.current.reachedStart).toBe(false);
    });
  });

  // ── PERF-2: the refresh defect ────────────────────────────────────────────────
  // The chart used to stay blank until the NEXT candle close (≤15 min on M15, up to
  // a day on D1) or a manual browser refresh, because a transient candle-feed
  // failure had no independent recovery: useCandles only re-fired on a combo change
  // or a candleCloseTs advance. These tests pin the fix — self-healing WITHOUT a
  // candle close and WITHOUT reloading the page.
  describe('PERF-2 — recovers a blank chart without a candle close or page reload', () => {
    it('auto-retries a TRANSIENT failure and heals itself (no candleCloseTs change)', async () => {
      vi.useFakeTimers();
      try {
        mockFetchWindow
          .mockRejectedValueOnce(new CandlesError(0, 'server unreachable', 'network'))
          .mockResolvedValue(win(SERIES));
        const { result } = renderHook(() =>
          useCandles('XAUUSD', 'M15', { source: 'live' }),
        );
        // Initial attempt fails → blank chart + error. No candle close happened.
        await act(async () => {
          await vi.advanceTimersByTimeAsync(0);
        });
        expect(mockFetchWindow).toHaveBeenCalledTimes(1);
        expect(result.current.candles).toBeNull();
        expect(result.current.error).not.toBeNull();
        // The bounded auto-retry fires after the backoff — candleCloseTs never moved.
        await act(async () => {
          await vi.advanceTimersByTimeAsync(1600);
        });
        expect(mockFetchWindow).toHaveBeenCalledTimes(2);
        expect(result.current.candles).toEqual(SERIES);
        expect(result.current.error).toBeNull();
      } finally {
        vi.useRealTimers();
      }
    });

    it('bounds the auto-retry to a finite number of attempts', async () => {
      vi.useFakeTimers();
      try {
        mockFetchWindow.mockRejectedValue(new CandlesError(0, 'timed out', 'timeout'));
        renderHook(() => useCandles('XAUUSD', 'M15', { source: 'live' }));
        // Step through each linear backoff (1500·1, 1500·2, 1500·3), flushing the
        // React re-render + effect between each so the next retry is scheduled.
        await act(async () => { await vi.advanceTimersByTimeAsync(0); });
        expect(mockFetchWindow).toHaveBeenCalledTimes(1); // initial
        await act(async () => { await vi.advanceTimersByTimeAsync(1600); });
        expect(mockFetchWindow).toHaveBeenCalledTimes(2); // retry 1
        await act(async () => { await vi.advanceTimersByTimeAsync(3100); });
        expect(mockFetchWindow).toHaveBeenCalledTimes(3); // retry 2
        await act(async () => { await vi.advanceTimersByTimeAsync(4600); });
        expect(mockFetchWindow).toHaveBeenCalledTimes(4); // retry 3 (the bound)
        // Past the bound: no further attempts, ever (never an unbounded loop).
        await act(async () => { await vi.advanceTimersByTimeAsync(20_000); });
        expect(mockFetchWindow).toHaveBeenCalledTimes(4);
      } finally {
        vi.useRealTimers();
      }
    });

    it('does NOT auto-retry a deterministic 404, but refresh() recovers it', async () => {
      mockFetchWindow
        .mockRejectedValueOnce(new CandlesError(404, 'no candles for combo', 'nodata'))
        .mockResolvedValue(win(SERIES));
      const { result } = renderHook(() =>
        useCandles('XAUUSD', 'M15', { source: 'live' }),
      );
      await waitFor(() => expect(result.current.error).not.toBeNull());
      expect(result.current.candles).toBeNull();
      // 404 is deterministic — retrying blindly is futile, so it is NOT auto-retried.
      expect(mockFetchWindow).toHaveBeenCalledTimes(1);
      // The user's manual "Réessayer" re-pulls WITHOUT a page reload.
      act(() => {
        result.current.refresh();
      });
      await waitFor(() => expect(result.current.candles).toEqual(SERIES));
      expect(result.current.error).toBeNull();
    });
  });
});
