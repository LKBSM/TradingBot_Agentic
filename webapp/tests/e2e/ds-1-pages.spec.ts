import { test, expect, type Page } from '@playwright/test';
import path from 'node:path';
import { dismissCookieBanner } from './utils';
import { mockAllApis } from './ds-mock';
import { SAMPLE_CALENDAR_EVENT_ID } from '../../lib/ds-samples';

/**
 * DS-1 — capture every REAL product page rendered with the frozen real-data
 * samples (via mockAllApis), at both target viewports, for Claude Design. No
 * backend, no source changes: the pages fetch `/api/*` as usual and the mock
 * answers with the fixtures.
 */

const OUT = path.resolve(__dirname, '../../../docs/audits/ds-1/pages');
const VIEWPORTS = [
  { tag: 'desktop-1280x800', width: 1280, height: 800 },
  { tag: 'mobile-390x844', width: 390, height: 844 },
];

// Deep-link /app and /zones to the combo our fixture describes (XAU/USD H4), so
// the header/selector agree with the mocked reading.
const PAGES = [
  { name: 'accueil', url: '/' },
  { name: 'app', url: '/app?instrument=XAUUSD&timeframe=H4', hasChart: true },
  { name: 'zones', url: '/zones?instrument=XAUUSD&timeframe=H4', hasChart: true },
  { name: 'scanner', url: '/scanner' },
  { name: 'scanner-decrire', url: '/scanner/decrire' },
  { name: 'actualites', url: '/actualites', ready: '.calm-grid' },
  { name: 'actualites-fiche', url: `/actualites/${SAMPLE_CALENDAR_EVENT_ID}`, ready: '.cal-page' },
];

async function settle(page: Page, opts: { hasChart?: boolean; ready?: string } = {}) {
  await dismissCookieBanner(page);
  // wait for the SubscriptionGate / any loading spinner to resolve
  await page.locator('.animate-spin').first().waitFor({ state: 'detached', timeout: 12_000 }).catch(() => {});
  if (opts.ready) {
    await page.locator(opts.ready).first().waitFor({ state: 'visible', timeout: 15_000 }).catch(() => {});
  }
  if (opts.hasChart) {
    // wait for the candles fetch + lightweight-charts to paint the canvas
    await page.locator('canvas').first().waitFor({ state: 'visible', timeout: 15_000 }).catch(() => {});
    await page.waitForTimeout(1_500);
  }
  await page.waitForTimeout(700); // let remaining paints finish
}

for (const vp of VIEWPORTS) {
  test.describe(`DS-1 pages — ${vp.tag}`, () => {
    for (const p of PAGES) {
      test(`${p.name}`, async ({ page }) => {
        test.setTimeout(120_000); // dev compiles each route on first hit
        await mockAllApis(page);
        await page.setViewportSize({ width: vp.width, height: vp.height });
        await page.goto(p.url, { waitUntil: 'domcontentloaded' });
        await settle(page, { hasChart: p.hasChart, ready: p.ready });
        // sanity: the app shell/body rendered something real (not a bare error)
        await expect(page.locator('body')).toBeVisible();
        await page.screenshot({ path: path.join(OUT, `${p.name}--${vp.tag}.png`), fullPage: true });
      });
    }
  });
}
