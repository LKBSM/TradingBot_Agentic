import { renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Network + mock-candle layers are stubbed so the hook is tested in isolation.
vi.mock('../api-client', () => ({
  fetchCandles: vi.fn(),
}));
vi.mock('@/lib/mockReadings', () => ({
  READING_DATA_SOURCE: 'live',
  getMockCandles: vi.fn(),
}));

import { useCandles } from '../hooks';
import { fetchCandles } from '../api-client';
import { getMockCandles } from '@/lib/mockReadings';

const mockFetchCandles = vi.mocked(fetchCandles);
const mockGetMockCandles = vi.mocked(getMockCandles);

const SERIES = [
  { time: 1, open: 1, high: 2, low: 0.5, close: 1.5 },
  { time: 2, open: 1.5, high: 2.5, low: 1, close: 2 },
];

beforeEach(() => {
  mockFetchCandles.mockReset();
  mockGetMockCandles.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('useCandles', () => {
  it('stays idle (no fetch) when the combo is null', () => {
    const { result } = renderHook(() => useCandles(null, null, { source: 'live' }));
    expect(result.current.candles).toBeNull();
    expect(mockFetchCandles).not.toHaveBeenCalled();
  });

  it('resolves locally in mock mode without any network call', () => {
    mockGetMockCandles.mockReturnValue(SERIES);
    const { result } = renderHook(() =>
      useCandles('XAUUSD', 'M15', { source: 'mock' }),
    );
    expect(result.current.candles).toEqual(SERIES);
    expect(mockFetchCandles).not.toHaveBeenCalled();
    expect(mockGetMockCandles).toHaveBeenCalledWith('XAUUSD', 'M15');
  });

  it('fetches the live feed and exposes the candles', async () => {
    mockFetchCandles.mockResolvedValue(SERIES);
    const { result } = renderHook(() =>
      useCandles('XAUUSD', 'M15', { source: 'live' }),
    );
    await waitFor(() => expect(result.current.candles).toEqual(SERIES));
    expect(result.current.error).toBeNull();
    expect(mockFetchCandles).toHaveBeenCalledWith(
      'XAUUSD',
      'M15',
      // Requests a few-hundred-bar window (cache-served, no extra provider call).
      expect.objectContaining({ signal: expect.any(AbortSignal), limit: 400 }),
    );
  });

  it('collapses an unavailable feed to null candles + an error', async () => {
    mockFetchCandles.mockRejectedValue(new Error('no candles cached'));
    const { result } = renderHook(() =>
      useCandles('EURUSD', 'H4', { source: 'live' }),
    );
    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.candles).toBeNull();
  });

  it('treats an empty live window as unavailable (null)', async () => {
    mockFetchCandles.mockResolvedValue([]);
    const { result } = renderHook(() =>
      useCandles('XAUUSD', 'H1', { source: 'live' }),
    );
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.candles).toBeNull();
  });

  it('re-fetches when candle_close_ts changes', async () => {
    mockFetchCandles.mockResolvedValue(SERIES);
    const { rerender } = renderHook(
      ({ ts }) => useCandles('XAUUSD', 'M15', { source: 'live', candleCloseTs: ts }),
      { initialProps: { ts: '2026-05-26T11:00:00+00:00' } },
    );
    await waitFor(() => expect(mockFetchCandles).toHaveBeenCalledTimes(1));
    rerender({ ts: '2026-05-26T11:15:00+00:00' });
    await waitFor(() => expect(mockFetchCandles).toHaveBeenCalledTimes(2));
  });
});
