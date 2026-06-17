import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { usePinnedCombos } from '@/lib/market-reading/pins';
import type { Combo } from '@/lib/market-reading/store';

const XAU_M15: Combo = { instrument: 'XAUUSD', timeframe: 'M15' };
const EUR_H1: Combo = { instrument: 'EURUSD', timeframe: 'H1' };
const STORAGE_KEY = 'mia.pinnedCombos.v1';

beforeEach(() => {
  window.localStorage.clear();
});

describe('usePinnedCombos', () => {
  it('starts empty', () => {
    const { result } = renderHook(() => usePinnedCombos());
    expect(result.current.pinned).toHaveLength(0);
    expect(result.current.isPinned(XAU_M15)).toBe(false);
  });

  it('toggles a pin on and persists to localStorage', async () => {
    const { result } = renderHook(() => usePinnedCombos());

    act(() => result.current.toggle(XAU_M15));

    await waitFor(() => expect(result.current.isPinned(XAU_M15)).toBe(true));
    expect(result.current.pinned).toEqual([XAU_M15]);
    expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY)!)).toEqual([
      'XAUUSD:M15',
    ]);
  });

  it('toggles a pin off again', async () => {
    const { result } = renderHook(() => usePinnedCombos());
    act(() => result.current.toggle(XAU_M15));
    await waitFor(() => expect(result.current.isPinned(XAU_M15)).toBe(true));

    act(() => result.current.toggle(XAU_M15));
    await waitFor(() => expect(result.current.isPinned(XAU_M15)).toBe(false));
    expect(result.current.pinned).toHaveLength(0);
  });

  it('hydrates pinned combos from existing localStorage', async () => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(['EURUSD:H1']));
    const { result } = renderHook(() => usePinnedCombos());
    await waitFor(() => expect(result.current.isPinned(EUR_H1)).toBe(true));
    expect(result.current.pinned).toEqual([EUR_H1]);
  });

  it('ignores stale / out-of-catalogue keys in storage', async () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(['BTCUSD:M15', 'XAUUSD:M15', 'garbage']),
    );
    const { result } = renderHook(() => usePinnedCombos());
    await waitFor(() => expect(result.current.pinned).toEqual([XAU_M15]));
  });

  it('keeps insertion order (most-recent pinned last)', async () => {
    const { result } = renderHook(() => usePinnedCombos());
    act(() => result.current.toggle(EUR_H1));
    act(() => result.current.toggle(XAU_M15));
    await waitFor(() => expect(result.current.pinned).toHaveLength(2));
    expect(result.current.pinned).toEqual([EUR_H1, XAU_M15]);
  });
});
