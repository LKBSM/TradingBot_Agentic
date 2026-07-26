/**
 * MC-1 — market status view derived from the SERVER's `reading.market_status`.
 *
 * The server (src/intelligence/market_calendar) is the single source of truth:
 * it combines the fact (age of the last closed candle) with the trading
 * calendar (weekend / holiday / daily break, per instrument, DST-safe). The
 * client never recomputes open/closed from its own clock — it only formats what
 * the server sent. When the payload is absent (static/landing fixtures) callers
 * fall back to the legacy client heuristic in ./session.
 */

import type { MarketState, MarketStatusPayload } from '@/types/market-reading';

export interface MarketStatusView {
  state: MarketState;
  /** Closed for a session reason (weekend / holiday / daily break). */
  isClosed: boolean;
  /** Calendar open but the feed stalled — "données en retard". */
  isLagged: boolean;
  /** Anything that is not a live, fresh feed → the "EN DIRECT" pulse is off. */
  isLive: boolean;
  lastCloseTs: string | null;
  nextOpenTs: string | null;
}

const CLOSED_STATES: ReadonlyArray<MarketState> = [
  'closed_weekend',
  'closed_holiday',
  'daily_break',
];

/** Build the view from the server payload, or null when none was sent. */
export function deriveMarketStatus(
  payload: MarketStatusPayload | null | undefined,
): MarketStatusView | null {
  if (!payload || !payload.state) return null;
  const state = payload.state;
  const isClosed = CLOSED_STATES.includes(state);
  const isLagged = state === 'data_lagged';
  return {
    state,
    isClosed,
    isLagged,
    isLive: state === 'open',
    lastCloseTs: payload.last_close_ts ?? null,
    nextOpenTs: payload.next_open_ts ?? null,
  };
}

/**
 * i18n key (under the `app.chart` namespace) for the badge label of a state.
 * `open` has no badge (the live indicator shows instead).
 */
export function badgeLabelKey(state: MarketState): string | null {
  switch (state) {
    case 'closed_weekend':
    case 'closed_holiday':
      return 'chart.marketClosed';
    case 'daily_break':
      return 'chart.marketPaused';
    case 'data_lagged':
      return 'chart.dataLagged';
    default:
      return null;
  }
}

/**
 * Format an ISO-8601 UTC instant as a localized weekday + time in New-York wall
 * clock (the reference the market hours are published in). Returns e.g.
 * "vendredi 17:00". The "(New York)" suffix lives in the i18n template, so this
 * is only the weekday+time. Returns null for a missing/invalid timestamp.
 */
export function formatNyTimestamp(
  iso: string | null | undefined,
  locale: string,
): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  try {
    return new Intl.DateTimeFormat(locale, {
      weekday: 'long',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
      timeZone: 'America/New_York',
    }).format(d);
  } catch {
    return null;
  }
}

/** i18n key for the badge tooltip/title of a state. */
export function badgeTitleKey(state: MarketState): string | null {
  switch (state) {
    case 'closed_weekend':
    case 'closed_holiday':
      return 'chart.marketClosedTitle';
    case 'daily_break':
      return 'chart.marketPausedTitle';
    case 'data_lagged':
      return 'chart.dataLaggedTitle';
    default:
      return null;
  }
}
