import type { Candle } from '@/types/market-reading';

/**
 * Descriptive daily change, derived purely from a closed-candle OHLC window.
 *
 * Why this exists (founder eval 2026-06-08): the reading header used
 * `header.close_price`, which is the last *closed* candle on the DISPLAYED
 * timeframe. On H1/H4 that candle can be up to an hour / four hours old, so the
 * same asset showed a different price on M15 vs H1/H4. We instead derive ONE
 * unified "last price" from the finest available timeframe (M15) — the freshest
 * closed price in the descriptive cache — identical whatever timeframe is shown.
 *
 * The percentage is a market FACT, never a forecast: `(last - referenceClose) /
 * referenceClose`, where `referenceClose` is the close of the previous UTC
 * trading day (TradingView-style daily change). Computed from the same closed
 * candles — no projection, no extra provider call.
 */
export interface DailyChange {
  /** Last closed price (close of the most recent candle in the window). */
  price: number;
  /** Epoch seconds of that last candle (UTCTimestamp). */
  priceTs: number;
  /** Close of the previous UTC day's last candle, or null if the window is too short. */
  referenceClose: number | null;
  /** Absolute change vs the reference, or null when no reference is available. */
  changeAbs: number | null;
  /** Fractional change vs the reference (e.g. -0.0322 for -3.22%), or null. */
  changePct: number | null;
}

/** UTC calendar-day key (YYYYMMDD as a number) for an epoch-seconds timestamp. */
function utcDayKey(epochSeconds: number): number {
  const d = new Date(epochSeconds * 1000);
  return d.getUTCFullYear() * 10_000 + (d.getUTCMonth() + 1) * 100 + d.getUTCDate();
}

/**
 * Compute the unified last price + descriptive daily change from an ascending
 * OHLC window. Returns null only when there are no candles at all.
 *
 * The reference is the close of the LAST candle belonging to the most recent
 * day STRICTLY before the latest candle's UTC day. Because weekends/holidays
 * simply have no candles, this naturally resolves to the previous *trading*
 * day's close (e.g. Friday's close on a Monday).
 */
export function computeDailyChange(candles: Candle[] | null | undefined): DailyChange | null {
  if (!candles || candles.length === 0) return null;

  const last = candles[candles.length - 1]!;
  const price = last.close;
  const priceTs = last.time;
  const lastDay = utcDayKey(last.time);

  let referenceClose: number | null = null;
  // Walk backwards to the first candle on an earlier UTC day → that day's close.
  for (let i = candles.length - 1; i >= 0; i -= 1) {
    if (utcDayKey(candles[i]!.time) < lastDay) {
      referenceClose = candles[i]!.close;
      break;
    }
  }

  if (referenceClose === null || referenceClose === 0) {
    return { price, priceTs, referenceClose, changeAbs: null, changePct: null };
  }

  const changeAbs = price - referenceClose;
  return {
    price,
    priceTs,
    referenceClose,
    changeAbs,
    changePct: changeAbs / referenceClose,
  };
}
