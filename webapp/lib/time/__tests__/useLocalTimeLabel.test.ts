import { renderHook, act } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useLocalTimeLabel } from '../useLocalTimeLabel';

/**
 * The hook must reflect the reader's CURRENT browser offset, and re-resolve it
 * when the tab regains focus/visibility so a mid-session timezone change (VPN,
 * OS timezone edit, travel, DST crossing) shows without a reload.
 */

// getTimezoneOffset returns minutes WEST of UTC (UTC-4 → +240), so utcOffsetLabel
// negates it. We drive a mutable value to simulate the OS timezone changing.
let offsetWestMinutes = 240; // start at UTC−4 (e.g. Québec EDT)
const spy = vi
  .spyOn(Date.prototype, 'getTimezoneOffset')
  .mockImplementation(() => offsetWestMinutes);

afterEach(() => {
  offsetWestMinutes = 240;
  spy.mockClear();
});

describe('useLocalTimeLabel', () => {
  it('resolves the browser offset on mount (client-only, no SSR mismatch)', () => {
    const { result } = renderHook(() => useLocalTimeLabel());
    // After mount effects run, the label reflects the current offset.
    expect(result.current).toBe('Heure locale · UTC−4');
  });

  it('re-resolves when the window regains focus (VPN / travel mid-session)', () => {
    const { result } = renderHook(() => useLocalTimeLabel());
    expect(result.current).toBe('Heure locale · UTC−4');

    // Reader switches OS timezone to UTC+2 while the tab is open, then returns.
    offsetWestMinutes = -120;
    act(() => {
      window.dispatchEvent(new Event('focus'));
    });
    expect(result.current).toBe('Heure locale · UTC+2');
  });

  it('re-resolves when the tab becomes visible again', () => {
    const { result } = renderHook(() => useLocalTimeLabel());
    expect(result.current).toBe('Heure locale · UTC−4');

    offsetWestMinutes = 0; // moved to UTC
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => 'visible',
    });
    act(() => {
      document.dispatchEvent(new Event('visibilitychange'));
    });
    expect(result.current).toBe('Heure locale · UTC');
  });

  it('ignores visibilitychange when the tab is being HIDDEN', () => {
    const { result } = renderHook(() => useLocalTimeLabel());
    expect(result.current).toBe('Heure locale · UTC−4');

    offsetWestMinutes = -330; // would be UTC+5:30 if it (wrongly) recomputed
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => 'hidden',
    });
    act(() => {
      document.dispatchEvent(new Event('visibilitychange'));
    });
    // Still the on-mount value — a hide event must not re-resolve.
    expect(result.current).toBe('Heure locale · UTC−4');
  });

  it('removes its listeners on unmount (no leak, no post-unmount update)', () => {
    const removeWin = vi.spyOn(window, 'removeEventListener');
    const removeDoc = vi.spyOn(document, 'removeEventListener');
    const { unmount } = renderHook(() => useLocalTimeLabel());
    unmount();
    expect(removeWin).toHaveBeenCalledWith('focus', expect.any(Function));
    expect(removeDoc).toHaveBeenCalledWith('visibilitychange', expect.any(Function));
    removeWin.mockRestore();
    removeDoc.mockRestore();
  });
});
