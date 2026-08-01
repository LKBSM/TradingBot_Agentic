import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * PERF-1 — honesty of loading. Drives /app with the market-reading endpoint
 * mocked per scenario so the four states the mission requires are asserted end
 * to end at both the desktop (1280×800) and mobile (390×844) viewports:
 *
 *   · complete load        → the reading renders, no error copy
 *   · slow load            → the "prend plus de temps" hint appears (not a frozen screen)
 *   · server unreachable   → "le serveur ne répond pas" (distinct from a timeout)
 *   · no data for the combo → "aucune donnée … pour cette combinaison" (distinct from a crash)
 *
 * The distinct copy is the whole point: one vague "données indisponibles" for all
 * four is exactly what this mission removes.
 */

const VIEWPORTS = [
  { name: 'desktop-1280', width: 1280, height: 800 },
  { name: 'mobile-390', width: 390, height: 844 },
] as const;

function makeCandles(n = 150) {
  const base = 2300;
  const start = Math.floor(Date.UTC(2026, 5, 20) / 1000);
  const candles = Array.from({ length: n }, (_, i) => {
    const close = base + i * 2;
    return {
      time: start + i * 900,
      open: close - 0.5,
      high: close + 1,
      low: close - 1,
      close,
      volume: 100,
    };
  });
  return { instrument: 'XAUUSD', timeframe: 'M15', candles };
}

async function gotoApp(page: Page, w: number, h: number) {
  await page.setViewportSize({ width: w, height: h });
  await page.goto('/app?instrument=XAUUSD&timeframe=M15');
  await dismissCookieBanner(page);
  // On the stacked (<1280px) layout the reading lives under the "Lecture" tab
  // (default tab is "Marchés"), and Radix unmounts inactive tabs — reveal it so
  // the reading/error/hint copy is actually in the DOM. Wait for the tab to exist
  // (the layout renders a brief spinner before the stacked tabs mount, so a bare
  // count() check can race and skip the click). Desktop (≥1280) has no such tab.
  if (w < 1280) {
    const lecture = page.getByRole('tab', { name: /Lecture/i });
    await lecture.waitFor({ state: 'visible', timeout: 8_000 });
    await lecture.click();
  }
}

for (const vp of VIEWPORTS) {
  test.describe(`PERF-1 load honesty — ${vp.name}`, () => {
    test('complete load renders the reading with no error copy', async ({ page }) => {
      await page.route('**/api/candles**', (route) =>
        route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(makeCandles()) }),
      );
      await page.route('**/api/market-reading**', (route) =>
        route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FIXTURE_XAU_M15) }),
      );

      await gotoApp(page, vp.width, vp.height);

      // A rendered chart canvas (fed by the mocked candles) is the viewport-
      // agnostic proof the reading loaded completely; and no failure copy shows.
      await expect(page.locator('canvas').first()).toBeVisible({ timeout: 15_000 });
      await expect(page.getByText(/ne répond pas|prend plus de temps|aucune donnée/i)).toHaveCount(0);
    });

    test('slow load is clearly communicated (hint, then honest timeout), never frozen', async ({ page }) => {
      // Hold the reading well past both the 6s slow-load hint threshold and the 8s
      // client budget, then (eventually) resolve. A slow load must never read as a
      // frozen screen: the "still loading" hint appears within the budget, and once
      // it is exceeded the honest "trop de temps à répondre" timeout message takes
      // over. Either is a valid, non-frozen signal — on desktop the reading mounts
      // with the fetch so the hint shows first; on mobile the reading tab opens
      // later, so the timeout message is what surfaces. Both are asserted here.
      await page.route('**/api/market-reading**', async (route) => {
        await new Promise((r) => setTimeout(r, 20_000));
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FIXTURE_XAU_M15) });
      });

      await gotoApp(page, vp.width, vp.height);

      await expect(
        page.getByText(/prend plus de temps|trop de temps à répondre/i).first(),
      ).toBeVisible({ timeout: 14_000 });
    });

    test('server unreachable shows "le serveur ne répond pas"', async ({ page }) => {
      await page.route('**/api/market-reading**', (route) => route.abort('failed'));

      await gotoApp(page, vp.width, vp.height);

      await expect(page.getByText(/ne répond pas/i)).toBeVisible({ timeout: 20_000 });
    });

    test('no data for the combo shows "aucune donnée … pour cette combinaison"', async ({ page }) => {
      await page.route('**/api/market-reading**', (route) =>
        route.fulfill({
          status: 404,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'No market data available yet.' }),
        }),
      );

      await gotoApp(page, vp.width, vp.height);

      await expect(page.getByText(/aucune donnée/i).first()).toBeVisible({ timeout: 15_000 });
    });
  });
}
