'use client';

import * as React from 'react';

/**
 * PROTOTYPE — opt-in live last-price stream (dev / free tier).
 *
 * Subscribes to the backend SSE `GET /api/live-price?instrument=…` (one shared
 * Twelve Data WebSocket behind the proxy; the browser never holds a key). Feeds
 * the PROVISIONAL, intra-candle zone-interaction overlay (FVG fill / OB touch).
 *
 * Gated by `NEXT_PUBLIC_LIVE_TICK` — OFF by default, so the app behaves exactly
 * as before (zone interaction refreshes only at candle close). When off, the
 * hook never opens a connection and always returns a null price.
 *
 * Descriptive only: the payload is the last traded price + its feed timestamp.
 * No prediction, no structure, never BOS/CHOCH.
 */

/** Whether the live-tick overlay is enabled on this build (env opt-in). */
export const LIVE_TICK_ENABLED: boolean = (() => {
  const v = process.env.NEXT_PUBLIC_LIVE_TICK;
  return v === '1' || v === 'true' || v === 'on';
})();

const ENDPOINT = '/api/live-price';

export interface UseLivePriceResult {
  /** Latest last-traded price for the instrument, or null when none yet. */
  price: number | null;
  /** Feed UNIX-epoch (seconds) of the latest tick, or null. */
  ts: number | null;
  /** True while the SSE connection is open. */
  connected: boolean;
}

const EMPTY: UseLivePriceResult = { price: null, ts: null, connected: false };

/**
 * Stream the live last price for `instrument` via SSE.
 *
 * Returns a null price when disabled (flag off), when `instrument` is null, or
 * before the first tick arrives. The connection is torn down on unmount and on
 * instrument change; EventSource handles transient reconnection. `enabled` lets
 * a caller force it off (e.g. mock mode / tests) regardless of the build flag.
 */
export function useLivePrice(
  instrument: string | null,
  options: { enabled?: boolean } = {},
): UseLivePriceResult {
  const enabled = (options.enabled ?? LIVE_TICK_ENABLED) && Boolean(instrument);

  const [state, setState] = React.useState<UseLivePriceResult>(EMPTY);

  React.useEffect(() => {
    if (!enabled || !instrument) {
      setState(EMPTY);
      return;
    }
    if (typeof window === 'undefined' || typeof EventSource === 'undefined') {
      return;
    }

    const url = `${ENDPOINT}?instrument=${encodeURIComponent(instrument)}`;
    const es = new EventSource(url);
    let closed = false;
    // Cap native auto-reconnect: after MAX_ERRORS consecutive failures without a
    // successful frame, stop retrying so a dead endpoint doesn't loop forever
    // (UI-16). Any good open/message resets the counter.
    const MAX_ERRORS = 5;
    let errorStreak = 0;

    es.onopen = () => {
      errorStreak = 0;
      if (!closed) setState((s) => ({ ...s, connected: true }));
    };
    es.onmessage = (ev: MessageEvent) => {
      if (closed) return;
      errorStreak = 0;
      try {
        const data = JSON.parse(ev.data) as {
          instrument?: string;
          price?: number;
          ts?: number;
        };
        if (data.instrument !== instrument) return;
        if (typeof data.price !== 'number' || !Number.isFinite(data.price)) return;
        setState({
          price: data.price,
          ts: typeof data.ts === 'number' ? data.ts : null,
          connected: true,
        });
      } catch {
        // Ignore a malformed frame — keep the last good price.
      }
    };
    es.onerror = () => {
      if (closed) return;
      setState((s) => ({ ...s, connected: false }));
      errorStreak += 1;
      if (errorStreak >= MAX_ERRORS) {
        // Give up rather than let EventSource reconnect indefinitely.
        closed = true;
        es.close();
      }
    };

    return () => {
      closed = true;
      es.close();
    };
  }, [enabled, instrument]);

  return state;
}
