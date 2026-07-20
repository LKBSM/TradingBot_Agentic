'use client';

import * as React from 'react';

/**
 * Descriptive "is this market open right now?" helper for the reading header /
 * chart. It answers a present FACT — never a forecast, and never a claim about
 * WHEN a closed market reopens.
 *
 * Two signals, combined so each covers the other's blind spot:
 *   · FX weekend calendar — spot FX and gold (XAU) do not trade from Friday
 *     ~22:00 UTC to Sunday ~22:00 UTC. Deterministic, no data needed.
 *   · Feed staleness — the freshest price/candle age. A live feed PROVES the
 *     market is open (overrides the calendar around the DST-shifted Sunday
 *     reopen); a feed that has been silent for hours means the session is
 *     closed for a reason the calendar can't know (a public holiday).
 *
 * Crypto (BTC/ETH…) trades 24/7 → never "closed".
 */

/** Instruments that trade 24/7 (crypto) — never reported as closed. */
export function isTwentyFourSevenMarket(instrument: string): boolean {
  return /BTC|ETH|USDT|USDC|crypto/i.test(instrument);
}

/**
 * True when `now` falls inside the FX / metals weekend close window: from Friday
 * ~22:00 UTC to Sunday ~22:00 UTC. Spot FX and gold do not trade across the
 * weekend. Purely descriptive — no claim about the exact reopening moment.
 */
export function isForexWeekend(now: Date): boolean {
  const day = now.getUTCDay(); // 0 = Sunday … 6 = Saturday
  const hour = now.getUTCHours();
  if (day === 6) return true; // all of Saturday
  if (day === 5 && hour >= 22) return true; // Friday from 22:00 UTC
  if (day === 0 && hour < 22) return true; // Sunday before 22:00 UTC
  return false;
}

/**
 * Staleness threshold (seconds). Beyond this the last known price is old enough
 * that the session is closed (weekend OR holiday). The header price is read from
 * the M15 window, so an OPEN weekday feed is always well under this (a candle
 * closes every 15 min); a weekend / holiday runs to many hours. 3 h leaves ample
 * margin over the 15-min cadence, so a transient provider hiccup won't flip the
 * badge, while a real holiday (a whole silent day) still does.
 */
export const MARKET_STALE_THRESHOLD_SEC = 3 * 60 * 60;

export interface MarketClosedOptions {
  /** Evaluation instant (defaults to the real clock). */
  now?: Date;
  /** Epoch seconds of the freshest known price / candle, or null if unknown. */
  priceTs?: number | null;
  /** Override the staleness threshold (seconds). */
  staleThresholdSec?: number;
}

/**
 * Descriptive check: is `instrument` closed at `now`? Pure + deterministic.
 *
 *   · Crypto → always open (24/7).
 *   · A FRESH feed (price age below the threshold) → OPEN, whatever the calendar
 *     says (guards the DST edges around the Sunday reopen).
 *   · Otherwise CLOSED when it is the FX weekend OR the feed is stale (holiday).
 *
 * Never predicts the reopening.
 */
export function isMarketClosed(
  instrument: string,
  options: MarketClosedOptions = {},
): boolean {
  if (isTwentyFourSevenMarket(instrument)) return false;

  const now = options.now ?? new Date();
  const nowSec = Math.floor(now.getTime() / 1000);
  const threshold = options.staleThresholdSec ?? MARKET_STALE_THRESHOLD_SEC;
  const priceTs = options.priceTs ?? null;
  const ageSec = priceTs != null ? nowSec - priceTs : null;

  // A fresh feed proves the market is open — this wins over the calendar so a
  // slightly-off weekend boundary (DST) never contradicts live data.
  if (ageSec != null && ageSec >= 0 && ageSec < threshold) return false;

  const stale = ageSec != null && ageSec >= threshold;
  return isForexWeekend(now) || stale;
}

/**
 * Live "is the market closed" state for `instrument`, refreshed on a light
 * interval so the badge flips within ~1 min of a session boundary. SSR-safe:
 * returns `false` until mounted (matching the neutral server render), then the
 * effect computes the real value on the client where the clock is available.
 */
export function useMarketClosed(
  instrument: string | null,
  priceTs: number | null,
  intervalMs = 60_000,
): boolean {
  const [closed, setClosed] = React.useState(false);

  React.useEffect(() => {
    if (!instrument) {
      setClosed(false);
      return;
    }
    const compute = () =>
      setClosed(isMarketClosed(instrument, { priceTs: priceTs ?? null }));
    compute();
    const id = window.setInterval(compute, intervalMs);
    return () => window.clearInterval(id);
  }, [instrument, priceTs, intervalMs]);

  return closed;
}
