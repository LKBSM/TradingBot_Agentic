import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * PERF-2 — the refresh defect. The candle feed (the chart) used to have NO
 * independent recovery: a transient failure left the chart blank until the next
 * candle close (≤15 min on M15, up to a day on D1) or a MANUAL browser refresh.
 * These scenarios prove the chart now heals itself WITHOUT a page reload — the
 * whole point of the mission ("une page qui ne s'affiche pas est pire qu'une page
 * lente"). Asserted at desktop (1280×800) and mobile (390×844).
 *
 * The reading endpoint always succeeds here; only the CHART feed (limit=400)
 * fails, so the failure is isolated to the candles. `page.reload()` is never
 * called — a visible chart at the end therefore proves recovery without a reload.
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

const okCandles = (route: import('@playwright/test').Route) =>
  route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(makeCandles()),
  });

async function gotoApp(page: Page, w: number, h: number) {
  await page.setViewportSize({ width: w, height: h });
  await page.route('**/api/market-reading**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(FIXTURE_XAU_M15),
    }),
  );
  await page.goto('/app?instrument=XAUUSD&timeframe=M15');
  await dismissCookieBanner(page);
  if (w < 1280) {
    const lecture = page.getByRole('tab', { name: /Lecture/i });
    await lecture.waitFor({ state: 'visible', timeout: 8_000 });
    await lecture.click();
  }
}

for (const vp of VIEWPORTS) {
  test.describe(`PERF-2 chart recovery — ${vp.name}`, () => {
    test('a TRANSIENT candle failure heals itself — no page reload', async ({ page }) => {
      // The chart feed (limit=400) fails on its first two attempts (the cold-load
      // pair) with a transient 503, then succeeds. The bounded auto-retry must
      // re-pull and paint the chart on its own — candleCloseTs never advanced.
      let chartCalls = 0;
      await page.route('**/api/candles**', (route) => {
        const isChart = route.request().url().includes('limit=400');
        if (isChart) {
          chartCalls += 1;
          if (chartCalls <= 2) {
            return route.fulfill({
              status: 503,
              contentType: 'application/json',
              body: JSON.stringify({ detail: 'candle store warming up' }),
            });
          }
        }
        return okCandles(route);
      });

      await gotoApp(page, vp.width, vp.height);

      // Self-heals: the canvas appears without any reload (auto-retry backoff is
      // ~1.5 s·n; allow generous slack for CI).
      await expect(page.locator('canvas').first()).toBeVisible({ timeout: 20_000 });
    });

    test('a no-data chart recovers via "Réessayer" — no page reload', async ({ page }) => {
      // A deterministic 404 is NOT auto-retried (retrying blindly is futile); the
      // placeholder names the cause and offers a manual retry. Clicking it re-pulls
      // WITHOUT reloading the page, and the chart paints.
      // On bascule sur un DRAPEAU, pas sur un compteur d'appels.
      //
      // Le compteur (`chartCalls <= 2`) supposait DEUX requêtes au chargement :
      // c'est vrai sous `next dev`, où `reactStrictMode` monte les composants
      // deux fois et double l'effet — mais faux dans un build de production,
      // qui n'en émet qu'une. En production le clic « Réessayer » tombait donc
      // sur le 2ᵉ appel, encore en 404, et le graphique ne peignait jamais.
      // D'où un test vert en local et rouge en intégration continue.
      //
      // Le drapeau exprime l'intention réelle — « la source échoue, puis elle
      // guérit » — sans rien supposer du nombre de montages de React.
      let chartFails = true;
      await page.route('**/api/candles**', (route) => {
        const isChart = route.request().url().includes('limit=400');
        if (isChart && chartFails) {
          return route.fulfill({
            status: 404,
            contentType: 'application/json',
            body: JSON.stringify({ detail: 'no candles for this combo yet' }),
          });
        }
        return okCandles(route);
      });

      await gotoApp(page, vp.width, vp.height);

      // Honest placeholder + a retry that does NOT reload the page.
      //
      // On vise le bouton DU GRAPHIQUE par son marqueur, pas « le dernier
      // Réessayer de la page » : à 1280 l'aperçu du calendrier en affiche un
      // second (`.cal-retry`), et `.last()` cliquait sur celui-là — le graphique
      // n'était jamais relancé, aucun canvas n'apparaissait. À 390 l'aperçu n'est
      // pas rendu, d'où un test vert à un viewport et rouge à l'autre.
      const retry = page.getByTestId('chart-retry');
      await expect(retry).toBeVisible({ timeout: 15_000 });
      chartFails = false; // la source guérit juste avant le clic
      await retry.click();

      await expect(page.locator('canvas').first()).toBeVisible({ timeout: 15_000 });
    });
  });
}
