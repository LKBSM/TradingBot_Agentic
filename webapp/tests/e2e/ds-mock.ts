import type { Page } from '@playwright/test';
import {
  SAMPLE_READING_XAU_H4,
  SAMPLE_CANDLES_XAU_H4,
  SAMPLE_SCAN_RESPONSE,
  SAMPLE_SCAN_CONFIG,
  SAMPLE_CALENDAR_MONTH,
  SAMPLE_CALENDAR_EVENT,
  SAMPLE_CALENDAR_MEASURES,
  SAMPLE_PALETTE_RESPONSE,
} from '../../lib/ds-samples';

/**
 * DS-1 — serves EVERY /api/* endpoint the product pages hit, from the frozen
 * real-data samples, so the REAL pages (/app, /zones, /scanner, /actualites,
 * home) render with no backend and can be captured for Claude Design — without
 * touching a line of the pages.
 *
 * TIMESTAMP RECAL: the fixtures are dated in the past (their real capture date),
 * which the app would read as stale → "marché fermé", empty chart, empty current
 * month. So every ISO timestamp (and each candle `time`) is shifted by a single
 * delta that lands the latest candle on ~now. Only ABSOLUTE dates move; every
 * value (OHLC, levels, zones, events) is the real product data, untouched.
 *
 * Route precedence in Playwright: the handler added LAST wins — so we register
 * least-specific first (catch-all, then `/calendar`, then `/calendar/month`, …).
 */

const ISO_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/;

/** Delta (seconds) that moves the latest sample candle to ~now (minus a small
 *  margin so the very last bar reads as a just-closed candle, not the future). */
function deltaSeconds(): number {
  const latest = SAMPLE_CANDLES_XAU_H4[SAMPLE_CANDLES_XAU_H4.length - 1]!.time;
  const nowSec = Math.floor(Date.now() / 1000) - 4 * 3600; // ~1 H4 bar ago
  return nowSec - latest;
}

/** Deep-copy a value, shifting every ISO-timestamp string and any `time`
 *  (epoch-seconds) field by delta. Prose/other strings are left untouched. */
function shift<T>(value: T, delta: number, key?: string): T {
  if (typeof value === 'string') {
    if (ISO_RE.test(value) && value.length < 40) {
      return new Date(Date.parse(value) + delta * 1000).toISOString() as unknown as T;
    }
    return value;
  }
  if (typeof value === 'number') {
    // candle epoch-seconds live under the `time` key
    return (key === 'time' ? value + delta : value) as unknown as T;
  }
  if (Array.isArray(value)) return value.map((v) => shift(v, delta)) as unknown as T;
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) out[k] = shift(v, delta, k);
    return out as unknown as T;
  }
  return value;
}

const json = (body: unknown) => ({
  status: 200,
  contentType: 'application/json',
  body: JSON.stringify(body),
});

const FULL_ACCESS = {
  authenticated: true,
  gate_enforced: false,
  beta_lockdown: false,
  must_login: false,
  is_owner: true,
  has_access: true,
  subscription_required: false,
};

export async function mockAllApis(page: Page): Promise<void> {
  const d = deltaSeconds();
  const reading = shift(SAMPLE_READING_XAU_H4, d);
  const candles = shift([...SAMPLE_CANDLES_XAU_H4], d);
  // MC-1: the reading's own market_status is authoritative for the chart/header.
  // Mark it OPEN + fresh so the real chart paints the candles (instead of the
  // "marché fermé" empty state it shows for past-dated data against the clock).
  reading.market_status = {
    state: 'open', reason: '', instrument: 'XAUUSD', timeframe: 'H4',
    last_close_ts: reading.header.candle_close_ts, next_open_ts: null, bars_behind: 0,
    continuous: false,
  };
  const calendarMonth = shift(SAMPLE_CALENDAR_MONTH, d);
  const calendarEvent = shift(SAMPLE_CALENDAR_EVENT, d);
  // fetchCalendarEvent expects a CalendarResponse whose `events` holds the match.
  const calendarEventResponse = { ...calendarMonth, events: [calendarEvent] };
  const lastClose = reading.header.candle_close_ts;

  // 0 — catch-all so nothing ever reaches the real backend (empty 200).
  await page.route('**/api/**', (r) => r.fulfill(json({})));

  // 1 — calendar (general → month), then the more specific variants.
  await page.route('**/api/calendar**', (r) => r.fulfill(json(calendarMonth)));
  await page.route('**/api/calendar/month**', (r) => r.fulfill(json(calendarMonth)));
  await page.route('**/api/calendar/event/**', (r) => r.fulfill(json(calendarEventResponse)));
  await page.route('**/api/publications/**', (r) => r.fulfill(json(SAMPLE_CALENDAR_MEASURES)));

  // 2 — market reading surfaces.
  await page.route('**/api/candles**', (r) =>
    r.fulfill(json({ instrument: 'XAUUSD', timeframe: 'H4', candles, has_more_history: false })),
  );
  await page.route('**/api/market-reading**', (r) => r.fulfill(json(reading)));
  await page.route('**/api/market-status**', (r) =>
    r.fulfill(json({
      state: 'open', reason: '', instrument: 'XAUUSD', timeframe: 'H4',
      last_close_ts: lastClose, next_open_ts: null, bars_behind: 0, continuous: false,
    })),
  );

  // 3 — scanner (general scan, then palette + translate).
  await page.route('**/api/conditions-scan**', (r) => r.fulfill(json(SAMPLE_SCAN_RESPONSE)));
  await page.route('**/api/conditions-scan/palette**', (r) => r.fulfill(json(SAMPLE_PALETTE_RESPONSE)));
  await page.route('**/api/scanner/translate**', (r) => r.fulfill(json(SAMPLE_SCAN_CONFIG)));

  // 4 — streams we never want to hang on (live tick / chat).
  await page.route('**/api/live-price**', (r) => r.abort());
  await page.route('**/api/chatbot/**', (r) => r.abort());

  // 5 — the access gate (most specific; wins over the catch-all).
  await page.route('**/api/access/me', (r) => r.fulfill(json(FULL_ACCESS)));
}
