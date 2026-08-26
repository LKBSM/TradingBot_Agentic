import { test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * UI-3 text-density pass — before/after capture harness.
 *
 * Captures the three DAILY surfaces UI-3 touched (/app, /scanner/decrire, /zones)
 * at both viewports, fr + en, so the founder can judge the density change by eye
 * (the mission's primary deliverable). Reading endpoints are mocked (the dev
 * server proxies /api to an absent backend under test).
 *
 * Phase is read from UI3_PHASE (before | after); default 'after'. Run once on the
 * working tree (after), then `git stash` the source edits and run again (before).
 */
const PHASE = process.env.UI3_PHASE === 'before' ? 'before' : 'after';
const DIR = `../docs/audits/ui-3-shots/${PHASE}`;

const START = Math.floor(Date.UTC(2026, 0, 1) / 1000);
const CANDLES = Array.from({ length: 300 }, (_, i) => {
  const close = 2000 + i;
  return { time: START + i * 900, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
});

async function mockReading(page: Page) {
  await page.route('**/api/candles**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ instrument: 'XAUUSD', timeframe: 'M15', candles: CANDLES, has_more_history: false }),
    }),
  );
  await page.route('**/api/market-reading**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FIXTURE_XAU_M15) }),
  );
  await page.route('**/api/market-status**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) }),
  );
}

const VIEWPORTS = [
  { w: 1280, h: 800, tag: 'desktop' },
  { w: 390, h: 844, tag: 'mobile' },
] as const;
const LOCALES = ['fr', 'en'] as const;

for (const vp of VIEWPORTS) {
  for (const loc of LOCALES) {
    test.describe(`ui-3 ${vp.tag} ${loc}`, () => {
      test.use({ viewport: { width: vp.w, height: vp.h } });

      test(`/app ${vp.tag} ${loc}`, async ({ page }) => {
        test.setTimeout(90_000);
        await mockReading(page);
        await page.goto(`/${loc}/app?instrument=XAUUSD&timeframe=M15`, { waitUntil: 'domcontentloaded' });
        await dismissCookieBanner(page);
        await page.waitForTimeout(2500);
        await page.screenshot({ path: `${DIR}/app-${vp.tag}-${loc}.png`, fullPage: true });
      });

      test(`/scanner/decrire ${vp.tag} ${loc}`, async ({ page }) => {
        await page.goto(`/${loc}/scanner/decrire`, { waitUntil: 'domcontentloaded' });
        await dismissCookieBanner(page);
        await page.waitForTimeout(1200);
        await page.screenshot({ path: `${DIR}/decrire-${vp.tag}-${loc}.png`, fullPage: true });
      });

      test(`/zones ${vp.tag} ${loc}`, async ({ page }) => {
        test.setTimeout(90_000);
        await mockReading(page);
        await page.goto(`/${loc}/zones?instrument=XAUUSD&timeframe=M15`, { waitUntil: 'domcontentloaded' });
        await dismissCookieBanner(page);
        await page.locator('[data-testid="distance-line"]').first().waitFor({ state: 'visible', timeout: 60_000 }).catch(() => {});
        await page.waitForTimeout(1500);
        await page.screenshot({ path: `${DIR}/zones-${vp.tag}-${loc}.png`, fullPage: true });
      });
    });
  }
}
