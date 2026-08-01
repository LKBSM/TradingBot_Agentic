import { renderHook, waitFor, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useMarketReading, __resetReadingRetention } from '../hooks';
import { FIXTURE_EUR_H1, FIXTURE_XAU_M15 } from '../fixtures';

// Mock the network layer — hooks are tested against fetchMarketReading's contract.
vi.mock('../api-client', () => ({
  fetchMarketReading: vi.fn(),
}));

import { fetchMarketReading } from '../api-client';

const mockFetch = vi.mocked(fetchMarketReading);

beforeEach(() => {
  mockFetch.mockReset();
  // The retention caches are module state shared across tests in this file —
  // clear them so each test's initial-load assertions start from a cold cache.
  __resetReadingRetention();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('useMarketReading', () => {
  it('stays idle (no request) when instrument/timeframe is null', () => {
    const { result } = renderHook(() => useMarketReading(null, null));
    expect(result.current.data).toBeNull();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('loads data on mount and clears the loading flag', async () => {
    mockFetch.mockResolvedValue(FIXTURE_XAU_M15);
    const { result } = renderHook(() => useMarketReading('XAUUSD', 'M15', { source: 'live' }));

    // Initial load flips isLoading on.
    expect(result.current.isLoading).toBe(true);

    await waitFor(() => expect(result.current.data).not.toBeNull());
    expect(result.current.data?.header.instrument).toBe('XAUUSD');
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(mockFetch).toHaveBeenCalledWith(
      'XAUUSD',
      'M15',
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it('surfaces an error when the fetch rejects', async () => {
    mockFetch.mockRejectedValue(new Error('503 unavailable'));
    const { result } = renderHook(() => useMarketReading('XAUUSD', 'M15', { source: 'live' }));

    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.error?.message).toContain('503');
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeNull();
  });

  it('refresh() re-fetches while keeping the previous reading visible', async () => {
    mockFetch.mockResolvedValue(FIXTURE_XAU_M15);
    const { result } = renderHook(() => useMarketReading('XAUUSD', 'M15', { source: 'live' }));
    await waitFor(() => expect(result.current.data).not.toBeNull());
    expect(mockFetch).toHaveBeenCalledTimes(1);

    act(() => result.current.refresh());

    // Stale data stays visible during the refresh; isRefreshing (not isLoading).
    expect(result.current.data).not.toBeNull();
    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(result.current.isRefreshing).toBe(false));
  });

  it('reloads (blank + load) when the combo changes', async () => {
    mockFetch.mockImplementation(async (instrument: string) =>
      instrument === 'XAUUSD' ? FIXTURE_XAU_M15 : FIXTURE_EUR_H1,
    );
    const { result, rerender } = renderHook(
      ({ i, t }: { i: string; t: string }) => useMarketReading(i, t, { source: 'live' }),
      { initialProps: { i: 'XAUUSD', t: 'M15' } },
    );
    await waitFor(() =>
      expect(result.current.data?.header.instrument).toBe('XAUUSD'),
    );

    rerender({ i: 'EURUSD', t: 'H1' });

    await waitFor(() =>
      expect(result.current.data?.header.instrument).toBe('EURUSD'),
    );
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it('PERF-1: revisiting a cached combo shows it instantly (isRefreshing, not isLoading)', async () => {
    mockFetch.mockResolvedValue(FIXTURE_XAU_M15);

    // First visit populates the retention cache.
    const first = renderHook(() => useMarketReading('XAUUSD', 'M15', { source: 'live' }));
    await waitFor(() => expect(first.result.current.data?.header.instrument).toBe('XAUUSD'));
    first.unmount();

    // Revisit (TF switch / return from another page): the cached reading shows on
    // the FIRST synchronous render — no blank skeleton — and a background refresh
    // (isRefreshing, the honesty signal) revalidates it.
    const second = renderHook(() => useMarketReading('XAUUSD', 'M15', { source: 'live' }));
    expect(second.result.current.data?.header.instrument).toBe('XAUUSD');
    expect(second.result.current.isLoading).toBe(false);
    expect(second.result.current.isRefreshing).toBe(true);

    await waitFor(() => expect(second.result.current.isRefreshing).toBe(false));
    expect(mockFetch).toHaveBeenCalledTimes(2); // still revalidated, never served blind
  });

  it('PERF-1: a first, uncached visit still blanks + shows the skeleton', async () => {
    mockFetch.mockResolvedValue(FIXTURE_XAU_M15);
    const { result } = renderHook(() => useMarketReading('XAUUSD', 'M15', { source: 'live' }));
    // Cold cache → honest skeleton, no phantom data.
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeNull();
    await waitFor(() => expect(result.current.data?.header.instrument).toBe('XAUUSD'));
  });
});
