import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Network + mock-candle layers are stubbed so the hook is tested in isolation.
// Keep the real error classes (CandlesError) — the hook references them at module
// scope (isTransientCandlesError) and for `instanceof` reason checks.
vi.mock('../api-client', async (importActual) => {
  const actual = await importActual<typeof import('../api-client')>();
  return { ...actual, fetchCandlesPage: vi.fn() };
});
vi.mock('@/lib/mockReadings', () => ({
  READING_DATA_SOURCE: 'live',
  getMockCandles: vi.fn(),
}));

import { useCandles, __resetReadingRetention } from '../hooks';
import { CandlesError, fetchCandlesPage } from '../api-client';
import { getMockCandles } from '@/lib/mockReadings';

const mockFetchCandlesPage = vi.mocked(fetchCandlesPage);
const mockGetMockCandles = vi.mocked(getMockCandles);

const SERIES = [
  { time: 1, open: 1, high: 2, low: 0.5, close: 1.5 },
  { time: 2, open: 1.5, high: 2.5, low: 1, close: 2 },
];

beforeEach(() => {
  mockFetchCandlesPage.mockReset();
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
    expect(mockFetchCandlesPage).not.toHaveBeenCalled();
  });

  it('resolves locally in mock mode without any network call', () => {
    mockGetMockCandles.mockReturnValue(SERIES);
    const { result } = renderHook(() =>
      useCandles('XAUUSD', 'M15', { source: 'mock' }),
    );
    expect(result.current.candles).toEqual(SERIES);
    expect(mockFetchCandlesPage).not.toHaveBeenCalled();
    expect(mockGetMockCandles).toHaveBeenCalledWith('XAUUSD', 'M15');
  });

  it('fetches the live feed and exposes the candles', async () => {
    mockFetchCandlesPage.mockResolvedValue({ candles: SERIES, hasMore: false });
    const { result } = renderHook(() =>
      useCandles('XAUUSD', 'M15', { source: 'live' }),
    );
    await waitFor(() => expect(result.current.candles).toEqual(SERIES));
    expect(result.current.error).toBeNull();
    expect(mockFetchCandlesPage).toHaveBeenCalledWith(
      'XAUUSD',
      'M15',
      // Requests a few-hundred-bar window (cache-served, no extra provider call).
      expect.objectContaining({ signal: expect.any(AbortSignal), limit: 400 }),
    );
  });

  it('collapses an unavailable feed to null candles + an error', async () => {
    mockFetchCandlesPage.mockRejectedValue(new Error('no candles cached'));
    const { result } = renderHook(() =>
      useCandles('EURUSD', 'H4', { source: 'live' }),
    );
    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.candles).toBeNull();
  });

  it('treats an empty live window as unavailable (null)', async () => {
    mockFetchCandlesPage.mockResolvedValue({ candles: [], hasMore: false });
    const { result } = renderHook(() =>
      useCandles('XAUUSD', 'H1', { source: 'live' }),
    );
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.candles).toBeNull();
  });

  it('re-fetches when candle_close_ts changes', async () => {
    mockFetchCandlesPage.mockResolvedValue({ candles: SERIES, hasMore: false });
    const { rerender } = renderHook(
      ({ ts }) => useCandles('XAUUSD', 'M15', { source: 'live', candleCloseTs: ts }),
      { initialProps: { ts: '2026-05-26T11:00:00+00:00' } },
    );
    await waitFor(() => expect(mockFetchCandlesPage).toHaveBeenCalledTimes(1));
    rerender({ ts: '2026-05-26T11:15:00+00:00' });
    await waitFor(() => expect(mockFetchCandlesPage).toHaveBeenCalledTimes(2));
  });

  // ── CHART-1: history pagination on zoom-out ───────────────────────────────────
  describe('CHART-1 — loads older history without ever dropping shown candles', () => {
    const RECENT = [
      { time: 3, open: 1, high: 2, low: 0.5, close: 1.5 },
      { time: 4, open: 1.5, high: 2.5, low: 1, close: 2 },
    ];
    const OLDER = [
      { time: 1, open: 1, high: 2, low: 0.5, close: 1.5 },
      { time: 2, open: 1.5, high: 2.5, low: 1, close: 2 },
    ];

    it('prepends the previous page, keeps existing bars, and updates the floor', async () => {
      mockFetchCandlesPage.mockImplementation(async (_i, _t, opts) =>
        opts?.before != null
          ? { candles: OLDER, hasMore: false }
          : { candles: RECENT, hasMore: true },
      );
      const { result } = renderHook(() => useCandles('XAUUSD', 'M15', { source: 'live' }));
      await waitFor(() => expect(result.current.candles).toEqual(RECENT));
      expect(result.current.hasMoreHistory).toBe(true);

      act(() => {
        result.current.loadOlder();
      });
      // The older page is merged BEFORE the shown bars — nothing vanished.
      await waitFor(() => expect(result.current.candles).toEqual([...OLDER, ...RECENT]));
      expect(result.current.hasMoreHistory).toBe(false); // reached the floor
      // loadOlder asked for candles strictly before the oldest shown (time=3).
      expect(mockFetchCandlesPage).toHaveBeenLastCalledWith(
        'XAUUSD',
        'M15',
        expect.objectContaining({ before: 3, limit: 400 }),
      );
    });

    it('is a no-op at the floor (hasMoreHistory false → no fetch)', async () => {
      mockFetchCandlesPage.mockResolvedValue({ candles: RECENT, hasMore: false });
      const { result } = renderHook(() => useCandles('XAUUSD', 'M15', { source: 'live' }));
      await waitFor(() => expect(result.current.candles).toEqual(RECENT));
      expect(result.current.hasMoreHistory).toBe(false);
      const callsBefore = mockFetchCandlesPage.mock.calls.length;
      act(() => {
        result.current.loadOlder();
      });
      expect(mockFetchCandlesPage.mock.calls.length).toBe(callsBefore); // no extra call
    });

    it('keeps the shown candles when a history load fails, and exposes the error', async () => {
      mockFetchCandlesPage.mockImplementation(async (_i, _t, opts) => {
        if (opts?.before != null) throw new CandlesError(0, 'network', 'network');
        return { candles: RECENT, hasMore: true };
      });
      const { result } = renderHook(() => useCandles('XAUUSD', 'M15', { source: 'live' }));
      await waitFor(() => expect(result.current.candles).toEqual(RECENT));
      act(() => {
        result.current.loadOlder();
      });
      await waitFor(() => expect(result.current.olderError).not.toBeNull());
      // The already-shown bars are untouched — a failed page never blanks the chart.
      expect(result.current.candles).toEqual(RECENT);
      expect(result.current.hasMoreHistory).toBe(true); // can retry
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
        mockFetchCandlesPage
          .mockRejectedValueOnce(new CandlesError(0, 'server unreachable', 'network'))
          .mockResolvedValue({ candles: SERIES, hasMore: false });
        const { result } = renderHook(() =>
          useCandles('XAUUSD', 'M15', { source: 'live' }),
        );
        // Initial attempt fails → blank chart + error. No candle close happened.
        await act(async () => {
          await vi.advanceTimersByTimeAsync(0);
        });
        expect(mockFetchCandlesPage).toHaveBeenCalledTimes(1);
        expect(result.current.candles).toBeNull();
        expect(result.current.error).not.toBeNull();
        // The bounded auto-retry fires after the backoff — candleCloseTs never moved.
        await act(async () => {
          await vi.advanceTimersByTimeAsync(1600);
        });
        expect(mockFetchCandlesPage).toHaveBeenCalledTimes(2);
        expect(result.current.candles).toEqual(SERIES);
        expect(result.current.error).toBeNull();
      } finally {
        vi.useRealTimers();
      }
    });

    it('bounds the auto-retry to a finite number of attempts', async () => {
      vi.useFakeTimers();
      try {
        mockFetchCandlesPage.mockRejectedValue(new CandlesError(0, 'timed out', 'timeout'));
        renderHook(() => useCandles('XAUUSD', 'M15', { source: 'live' }));
        // Step through each linear backoff (1500·1, 1500·2, 1500·3), flushing the
        // React re-render + effect between each so the next retry is scheduled.
        await act(async () => { await vi.advanceTimersByTimeAsync(0); });
        expect(mockFetchCandlesPage).toHaveBeenCalledTimes(1); // initial
        await act(async () => { await vi.advanceTimersByTimeAsync(1600); });
        expect(mockFetchCandlesPage).toHaveBeenCalledTimes(2); // retry 1
        await act(async () => { await vi.advanceTimersByTimeAsync(3100); });
        expect(mockFetchCandlesPage).toHaveBeenCalledTimes(3); // retry 2
        await act(async () => { await vi.advanceTimersByTimeAsync(4600); });
        expect(mockFetchCandlesPage).toHaveBeenCalledTimes(4); // retry 3 (the bound)
        // Past the bound: no further attempts, ever (never an unbounded loop).
        await act(async () => { await vi.advanceTimersByTimeAsync(20_000); });
        expect(mockFetchCandlesPage).toHaveBeenCalledTimes(4);
      } finally {
        vi.useRealTimers();
      }
    });

    it('does NOT auto-retry a deterministic 404, but refresh() recovers it', async () => {
      mockFetchCandlesPage
        .mockRejectedValueOnce(new CandlesError(404, 'no candles for combo', 'nodata'))
        .mockResolvedValue({ candles: SERIES, hasMore: false });
      const { result } = renderHook(() =>
        useCandles('XAUUSD', 'M15', { source: 'live' }),
      );
      await waitFor(() => expect(result.current.error).not.toBeNull());
      expect(result.current.candles).toBeNull();
      // 404 is deterministic — retrying blindly is futile, so it is NOT auto-retried.
      expect(mockFetchCandlesPage).toHaveBeenCalledTimes(1);
      // The user's manual "Réessayer" re-pulls WITHOUT a page reload.
      act(() => {
        result.current.refresh();
      });
      await waitFor(() => expect(result.current.candles).toEqual(SERIES));
      expect(result.current.error).toBeNull();
    });
  });
});
