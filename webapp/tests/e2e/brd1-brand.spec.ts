import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * Mission BRD-1 — brand presence in the connected surfaces.
 *
 * The wordmark ("MIA Markets") and its developed acronym ("Multi-asset
 * Intelligence Assistant") must be visible on every connected surface, on a
 * loading screen, and alongside an empty state — at 1280×800 and 390×844, in fr
 * and en. The brand is ADDED: it never replaces the loading indicator nor the
 * explicit empty-state message.
 *
 * Runs under the chromium-desktop project only; each test sets its own viewport
 * so the two required sizes are covered without doubling under the mobile
 * project (which would just override the viewport again). The desktop and mobile
 * layouts BOTH live in the DOM (CSS toggles them), so every brand assertion
 * targets the *visible* instance.
 */

const WORDMARK = 'MIA Markets';
const BASELINE = 'Multi-asset Intelligence Assistant';

const VIEWPORTS = [
  { name: 'desktop-1280', width: 1280, height: 800 },
  { name: 'mobile-390', width: 390, height: 844 },
] as const;

const LOCALES = [
  { name: 'fr', prefix: '' },
  { name: 'en', prefix: '/en' },
] as const;

const SURFACES = ['/app', '/scanner', '/zones', '/actualites', '/compte'] as const;

// A visible element containing `text` (both layouts are mounted; pick the shown one).
function visibleText(page: Page, text: string) {
  return page.getByText(text).and(page.locator(':visible')).first();
}

function makeCandles(n = 150) {
  const base = 2300;
  const start = Math.floor(Date.UTC(2026, 5, 20) / 1000);
  const candles = Array.from({ length: n }, (_, i) => {
    const close = base + i * 2;
    return { time: start + i * 900, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
  });
  return { instrument: 'XAUUSD', timeframe: 'M15', candles };
}

// Scanner empty state (no combo meets the conditions) — mocked, backend-free,
// reusing the sc1-scanner convention (seed a saved config + mock the scan).
const CONFIG_KEY = 'mia.conditionsConfig.v1';
const CONFIG = { logic: 'AND', conditions: [{ type: 'trend_is', trend: 'bearish' }, { type: 'price_in_ob' }] };
// Full context object — ScanResults reads context fields, so a partial one
// crashes the page (cf. sc1-scanner CTX).
const CTX = {
  trend: 'bearish', market_phase: 'trend', volatility_observed: 'normal',
  mtf_confluence: {}, mtf_trends: { h4: 'bearish', h1: 'bearish', m15: 'bearish' },
  bos: null, choch: null, active_order_blocks: 2, active_fair_value_gaps: 1,
  structural_range: { low: 4010, high: 4055 }, news_upcoming: [],
};
const NO_COMBO = {
  as_of: new Date().toISOString(),
  logic: 'AND',
  scanned: 1,
  matches: [
    {
      instrument: 'XAUUSD', timeframe: 'M15', candle_close_ts: new Date().toISOString(),
      close_price: 4029, matched: false, met_count: 1, total: 2, non_evaluable_count: 0,
      conditions_met: [{ type: 'trend_is', label: 'La tendance', met: true, detail: 'Baissière.' }],
      conditions_unmet: [{ type: 'price_in_ob', label: 'Prix dans un OB', met: false, detail: 'Hors OB.' }],
      conditions_non_evaluable: [], context_against: [], context: CTX, freshness: 'fresh', bars_behind: 0,
    },
  ],
  unavailable: [],
};

for (const vp of VIEWPORTS) {
  for (const loc of LOCALES) {
    test.describe(`BRD-1 brand — ${vp.name} · ${loc.name}`, () => {
      test.beforeEach(async ({ page }) => {
        await page.setViewportSize({ width: vp.width, height: vp.height });
      });

      test('every connected surface shows the wordmark and the acronym', async ({ page }) => {
        for (const surface of SURFACES) {
          await page.goto(`${loc.prefix}${surface}`);
          await dismissCookieBanner(page);
          await expect(visibleText(page, WORDMARK), `${surface} wordmark`).toBeVisible({ timeout: 10_000 });
          await expect(visibleText(page, BASELINE), `${surface} acronym`).toBeVisible();
        }
      });

      test('the loading screen carries the brand without hiding the loader', async ({ page }) => {
        await page.route('**/api/candles**', (route) =>
          route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(makeCandles()) }),
        );
        // Hold the reading long enough for the skeleton to stay while we assert.
        await page.route('**/api/market-reading**', async (route) => {
          await new Promise((r) => setTimeout(r, 6_000));
          route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FIXTURE_XAU_M15) });
        });

        await page.goto(`${loc.prefix}/app?instrument=XAUUSD&timeframe=M15`);
        await dismissCookieBanner(page);
        // On the stacked (<1280px) layout the reading lives under the second tab
        // (locale-agnostic: markets · reading · chat).
        if (vp.width < 1280) {
          const readingTab = page.getByRole('tab').nth(1);
          await readingTab.waitFor({ state: 'visible', timeout: 8_000 });
          await readingTab.click();
        }

        // Loading indicator intact…
        await expect(page.getByTestId('reading-skeleton')).toBeVisible({ timeout: 8_000 });
        // …and the brand is present on the same screen.
        await expect(visibleText(page, BASELINE)).toBeVisible();
      });

      test('an empty state keeps its explicit message and still shows the brand', async ({ page }) => {
        await page.route('**/api/access/me', (r) =>
          r.fulfill({ json: { has_full_access: true, entitlements: { instruments: [], timeframes: [] } } }),
        );
        await page.route('**/api/conditions-scan', (r) => r.fulfill({ json: NO_COMBO }));
        await page.addInitScript(
          (data: Record<string, string>) => {
            for (const [k, v] of Object.entries(data)) window.localStorage.setItem(k, v);
          },
          { [CONFIG_KEY]: JSON.stringify(CONFIG) },
        );

        await page.goto(`${loc.prefix}/scanner`);
        await dismissCookieBanner(page);
        // The explicit empty-state message stays as the priority content…
        await expect(page.getByTestId('scan-no-combo')).toBeVisible({ timeout: 12_000 });
        // …and the brand coexists (rail), never replacing it.
        await expect(visibleText(page, WORDMARK)).toBeVisible();
      });
    });
  }
}
