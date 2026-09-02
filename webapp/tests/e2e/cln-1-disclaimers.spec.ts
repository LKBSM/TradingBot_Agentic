import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * CLN-1 §5 — EXACTLY ONE educational/legal disclaimer per product page, at both
 * viewports. The single notice lives in the rail footer (desktop) and a
 * mobile-only footer (< 768px, where the rail is hidden); every inline
 * per-surface duplicate (the /app header line, the scanner combo note, the
 * /zones M.I.A note) was removed.
 *
 * The distinctive stem « Lecture algorithmique éducative » (fr) / « Educational
 * algorithmic reading » (en) is unique to that disclaimer — the chat/scanner
 * widget microcopy uses different wording, so it never collides. This guard
 * fails at ZERO (a page with no disclaimer) as loudly as at TWO (stacked).
 */

const FULL_ACCESS = {
  authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false,
  is_owner: true, has_access: true, subscription_required: false,
};

const D = 86400_000;
const iso = (ms: number) => new Date(Date.now() + ms).toISOString();

const START = Math.floor(Date.UTC(2026, 0, 1) / 1000);
const CANDLES = Array.from({ length: 60 }, (_, i) => {
  const close = 2000 + i;
  return { time: START + i * 900, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
});

const CALENDAR = {
  window_start: iso(-40 * D), window_end: iso(40 * D), generated_at: iso(0),
  coverage: { source: 'official', feed_start: null, feed_end: null, partial: false, last_success: {}, stale_sources: [] },
  attribution: [],
  events: [{
    event_id: 'bls:us_cpi:2026-08-12', source: 'bls', series_code: 'CUUR0000SA0', license_label: 'x',
    event: 'US_CPI', currency: 'USD', organism: 'Bureau of Labor Statistics', periodicity: 'monthly',
    scheduled_at: iso(12 * D), source_timezone: 'America/New_York', time_confirmed: true,
    markets: ['XAUUSD', 'EURUSD'], value_unit: '%',
    actual: null, actual_initial: 3.2, previous: 3.0, revised: false, revised_at: iso(-20 * D),
    actual_state: 'pending', refreshed_at: iso(0), value_series: [],
  }],
};

async function mockAll(page: Page) {
  await page.route('**/api/access/me', (r) => r.fulfill({ json: FULL_ACCESS }));
  await page.route('**/api/candles**', (r) =>
    r.fulfill({ json: { instrument: 'XAUUSD', timeframe: 'M15', candles: CANDLES, has_more_history: false } }),
  );
  await page.route('**/api/market-reading**', (r) => r.fulfill({ json: FIXTURE_XAU_M15 }));
  await page.route('**/api/market-status**', (r) => r.fulfill({ json: {} }));
  await page.route('**/api/publications/*/measures', (r) => r.fulfill({ json: { event_key: 'us_cpi', market: '', calm_before: null, structure_state: null, zone_lifecycle: null, return_to_calm: null } }));
  await page.route('**/api/calendar/event/*', (r) => r.fulfill({ json: CALENDAR }));
  await page.route('**/api/calendar*', (r) => r.fulfill({ json: CALENDAR }));
}

/** Count VISIBLE <p> elements whose text carries the page-disclaimer stem. */
async function visibleDisclaimers(page: Page, stem: string): Promise<number> {
  const loc = page.locator('p', { hasText: stem });
  let n = 0;
  for (const el of await loc.all()) if (await el.isVisible()) n += 1;
  return n;
}

// [route, localized prefix]. The `en` locale is served under /en (fr is default).
const PAGES = ['/app', '/scanner', '/scanner/decrire', '/zones?instrument=XAUUSD&timeframe=M15', '/actualites', '/compte'];

const LOCALES: Array<{ code: string; prefix: string; stem: string }> = [
  { code: 'fr', prefix: '', stem: 'Lecture algorithmique éducative' },
  { code: 'en', prefix: '/en', stem: 'Educational algorithmic reading' },
];

const VIEWPORTS: Array<{ name: string; width: number; height: number }> = [
  { name: 'desktop 1280×800', width: 1280, height: 800 },
  { name: 'mobile 390×844', width: 390, height: 844 },
];

for (const vp of VIEWPORTS) {
  test.describe(`CLN-1 §5 — one disclaimer per page @ ${vp.name}`, () => {
    test.use({ viewport: { width: vp.width, height: vp.height } });
    test.setTimeout(90_000);

    for (const loc of LOCALES) {
      for (const route of PAGES) {
        const url = `${loc.prefix}${route}`;
        test(`${loc.code} ${route} shows exactly one disclaimer`, async ({ page }) => {
          await mockAll(page);
          await page.goto(url, { waitUntil: 'domcontentloaded' });
          await dismissCookieBanner(page);
          // Let the shell (rail / mobile footer) render.
          await page.locator('.app-shell').first().waitFor({ state: 'attached', timeout: 60_000 });
          await expect
            .poll(() => visibleDisclaimers(page, loc.stem), { timeout: 20_000 })
            .toBe(1);
        });
      }
    }
  });
}
