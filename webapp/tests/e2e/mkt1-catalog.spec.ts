import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * MKT-1 — market-catalogue UX test, live capture at both viewports.
 *
 * ⚠️ THIS SPEC REQUIRES THE FLAG ON. The catalogue is inlined at build time, so
 * the server under test must have been built/started with
 * NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST=1. What it captures is therefore a
 * TEST state, never the default one — the flag is off in a standard build and
 * the companion spec mkt1-catalog-off.spec.ts asserts exactly that.
 *
 *   npm run build && npm start   (with the env var set)
 *   CI=1 E2E_BASE_URL=http://localhost:3123 npx playwright test mkt1-catalog
 */
const DIR = '../docs/audits/mkt-1-shots';

const START = Math.floor(Date.UTC(2026, 0, 1) / 1000);
const CANDLES = Array.from({ length: 300 }, (_, i) => {
  const close = 2000 + i;
  return { time: START + i * 900, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
});

/** Only the REAL market is mocked — a catalogue market must never be requested. */
async function mockReal(page: Page) {
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

for (const vp of VIEWPORTS) {
  test.describe(`mkt-1 ${vp.tag}`, () => {
    test.use({ viewport: { width: vp.w, height: vp.h } });

    test(`catalogue visible et groupé — ${vp.tag}`, async ({ page }) => {
      test.setTimeout(90_000);
      await mockReal(page);
      await page.goto('/fr/app?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
      await dismissCookieBanner(page);
      if (vp.tag === 'mobile') {
        // The rail is desktop-only; the mobile sidebar holds the same selector.
        const opener = page.getByRole('button', { name: /marchés/i }).first();
        if (await opener.isVisible().catch(() => false)) await opener.click();
      }
      await page.waitForTimeout(2000);

      // The mode must announce itself.
      await expect(page.getByTestId('mkt-uxtest-banner').first()).toBeVisible();
      await page.screenshot({ path: `${DIR}/catalogue-${vp.tag}.png`, fullPage: true });
    });

    test(`catégories dépliables — ${vp.tag}`, async ({ page }) => {
      test.setTimeout(90_000);
      await mockReal(page);
      await page.goto('/fr/app?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
      await dismissCookieBanner(page);
      if (vp.tag === 'mobile') {
        const opener = page.getByRole('button', { name: /marchés/i }).first();
        if (await opener.isVisible().catch(() => false)) await opener.click();
      }
      await page.waitForTimeout(1500);

      const head = page.getByTestId('mkt-group-head-crypto').first();
      await expect(head).toHaveAttribute('aria-expanded', 'false');
      await head.click();
      await expect(head).toHaveAttribute('aria-expanded', 'true');
      await page.waitForTimeout(400);
      await page.screenshot({ path: `${DIR}/categorie-ouverte-${vp.tag}.png`, fullPage: true });
    });

    test(`état vide honnête sur un marché sans donnée — ${vp.tag}`, async ({ page }) => {
      test.setTimeout(90_000);
      await mockReal(page);

      // Any request for a catalogue market is a failure of the whole mission —
      // record them rather than trusting that none happen.
      const leaked: string[] = [];
      page.on('request', (req) => {
        const url = req.url();
        if (/BTCUSD|XAGUSD|NAS100|USDTRY/.test(url)) leaked.push(url);
      });

      await page.goto('/fr/app?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
      await dismissCookieBanner(page);
      if (vp.tag === 'mobile') {
        const opener = page.getByRole('button', { name: /marchés/i }).first();
        if (await opener.isVisible().catch(() => false)) await opener.click();
      }
      await page.waitForTimeout(1500);

      await page.getByTestId('mkt-group-head-crypto').first().click();
      await page.getByText('Bitcoin (BTC/USD)').first().click();
      await page.waitForTimeout(1500);

      await expect(page.getByText('Pas encore disponible sur ce marché').first()).toBeVisible();
      // No substitute chart, no invented price.
      await expect(page.locator('canvas')).toHaveCount(0);
      expect(leaked, `requests leaked for an uncovered market:\n${leaked.join('\n')}`).toEqual([]);

      await page.screenshot({ path: `${DIR}/etat-vide-${vp.tag}.png`, fullPage: true });
    });

    test(`recherche à 100 entrées — ${vp.tag}`, async ({ page }) => {
      test.setTimeout(90_000);
      await mockReal(page);
      await page.goto('/fr/app?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
      await dismissCookieBanner(page);
      if (vp.tag === 'mobile') {
        const opener = page.getByRole('button', { name: /marchés/i }).first();
        if (await opener.isVisible().catch(() => false)) await opener.click();
      }
      await page.waitForTimeout(1500);

      const search = page.getByLabel(/Rechercher un marché/i).first();
      const t0 = Date.now();
      await search.fill('indices');
      await expect(page.getByTestId('mkt-group-index').first()).toBeVisible();
      const elapsed = Date.now() - t0;
      // Generous bound: guards a pathological filter, not a slow machine.
      expect(elapsed, `search took ${elapsed}ms`).toBeLessThan(4000);

      await page.screenshot({ path: `${DIR}/recherche-${vp.tag}.png`, fullPage: true });
    });
  });
}
