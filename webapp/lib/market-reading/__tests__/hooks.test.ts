import { renderHook, waitFor, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useMarketReading } from '../hooks';
import { FIXTURE_EUR_H1, FIXTURE_XAU_M15 } from '../fixtures';

// Mock the network layer — hooks are tested against fetchMarketReading's contract.
vi.mock('../api-client', () => ({
  fetchMarketReading: vi.fn(),
}));

import { fetchMarketReading } from '../api-client';

const mockFetch = vi.mocked(fetchMarketReading);

beforeEach(() => {
  mockFetch.mockReset();
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
});
