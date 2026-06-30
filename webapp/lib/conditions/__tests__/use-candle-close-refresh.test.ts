import { renderHook, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useCandleCloseRefresh } from '../use-candle-close-refresh';

const START = '2026-06-30T12:07:00Z'; // M15 boundary at 12:15
const M15_MS = 15 * 60_000;

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date(START));
});

afterEach(() => {
  vi.useRealTimers();
});

/** Advance the fake clock (both Date and timers) by `ms`, flushing effects. */
function advance(ms: number) {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
}

describe('useCandleCloseRefresh', () => {
  it('does NOT fire continuously — only at the candle close (+buffer)', () => {
    const onRefresh = vi.fn();
    renderHook(() =>
      useCandleCloseRefresh({
        timeframes: ['M15', 'H1', 'H4'],
        enabled: true,
        isScanning: false,
        onRefresh,
      }),
    );

    // One minute in: nothing yet (it's not a per-second/per-minute poll).
    advance(60_000);
    expect(onRefresh).not.toHaveBeenCalled();

    // Reach the 12:15 close + 15s backend buffer (8m15s from 12:07).
    advance(8 * 60_000 + 15_000);
    expect(onRefresh).toHaveBeenCalledTimes(1);

    // It re-arms for the FOLLOWING close (12:30) — one more fire per candle.
    advance(M15_MS);
    expect(onRefresh).toHaveBeenCalledTimes(2);
  });

  it('does nothing when disabled', () => {
    const onRefresh = vi.fn();
    renderHook(() =>
      useCandleCloseRefresh({
        timeframes: ['M15'],
        enabled: false,
        isScanning: false,
        onRefresh,
      }),
    );
    advance(2 * M15_MS);
    expect(onRefresh).not.toHaveBeenCalled();
  });

  it('skips the fire while a scan is already in flight (anti-avalanche)', () => {
    const onRefresh = vi.fn();
    renderHook(() =>
      useCandleCloseRefresh({
        timeframes: ['M15'],
        enabled: true,
        isScanning: true, // a scan is running the whole time
        onRefresh,
      }),
    );
    advance(2 * M15_MS);
    expect(onRefresh).not.toHaveBeenCalled();
  });

  it('does nothing with an empty timeframe set', () => {
    const onRefresh = vi.fn();
    renderHook(() =>
      useCandleCloseRefresh({
        timeframes: [],
        enabled: true,
        isScanning: false,
        onRefresh,
      }),
    );
    advance(2 * M15_MS);
    expect(onRefresh).not.toHaveBeenCalled();
  });
});
