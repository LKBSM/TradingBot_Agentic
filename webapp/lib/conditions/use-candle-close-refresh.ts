'use client';

import * as React from 'react';
import { nextCandleCloseMs } from './candle-clock';

/**
 * Auto-refresh the scan ALIGNED ON CANDLE CLOSES — not a per-second poll.
 *
 * Structural readings only change when a candle closes, so we schedule a single
 * timer for the soonest next close across the scanned timeframes, fire one
 * refresh just after it (plus a small buffer to let the backend regenerate the
 * just-closed candle), then re-arm for the following close. Each fire therefore
 * corresponds to a genuinely new candle — no redundant recomputation of the
 * same close.
 *
 * Anti-avalanche & graceful degradation:
 *  · never fires while a scan is already in flight (skips, re-arms);
 *  · pauses while the tab is hidden (one catch-up refresh on return);
 *  · a single timer at all times — no stacked requests.
 */

/** Let the backend's scheduler regenerate the just-closed candle first. */
const BACKEND_LAG_BUFFER_MS = 15_000;
/** Never schedule a near-instant fire (guards against clock skew at a boundary). */
const MIN_DELAY_MS = 5_000;

export interface UseCandleCloseRefreshOptions {
  /** Timeframes currently scanned (e.g. ['M15','H1','H4']). */
  timeframes: string[];
  /** Master switch (user preference). */
  enabled: boolean;
  /** True while a scan is running — used to skip overlapping fires. */
  isScanning: boolean;
  /** Trigger a re-scan. */
  onRefresh: () => void;
}

export function useCandleCloseRefresh({
  timeframes,
  enabled,
  isScanning,
  onRefresh,
}: UseCandleCloseRefreshOptions): void {
  // Keep the latest callbacks/flags in refs so the scheduling effect re-arms
  // ONLY when `enabled` or the timeframe set changes — not on every render.
  const onRefreshRef = React.useRef(onRefresh);
  onRefreshRef.current = onRefresh;
  const isScanningRef = React.useRef(isScanning);
  isScanningRef.current = isScanning;

  // Stable identity for the timeframe set (order-independent).
  const tfKey = React.useMemo(() => [...timeframes].sort().join(','), [timeframes]);

  React.useEffect(() => {
    if (!enabled || timeframes.length === 0 || typeof window === 'undefined') return;

    let timer: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;
    let missedWhileHidden = false;

    const rearm = () => {
      if (timer) clearTimeout(timer);
      if (cancelled) return;
      const now = Date.now();
      const boundary = nextCandleCloseMs(timeframes, now);
      if (boundary === null) return;
      const delay = Math.max(MIN_DELAY_MS, boundary + BACKEND_LAG_BUFFER_MS - now);
      timer = setTimeout(fire, delay);
    };

    const fire = () => {
      if (cancelled) return;
      if (document.visibilityState !== 'visible') {
        // Don't scan an invisible tab; remember to catch up on return.
        missedWhileHidden = true;
      } else if (!isScanningRef.current) {
        onRefreshRef.current();
      }
      rearm();
    };

    const onVisibility = () => {
      if (cancelled || document.visibilityState !== 'visible') return;
      if (missedWhileHidden && !isScanningRef.current) {
        missedWhileHidden = false;
        onRefreshRef.current();
      }
      rearm(); // realign to the next boundary from "now"
    };

    rearm();
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisibility);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, tfKey]);
}
