import { expect, test, type Page, type Route } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * CHART-1 — the chart renders with the full zoom/history control set at both
 * viewports, and its candle feed honours the paginating `before` + `has_more`
 * contract (mocked here). The zoom/pagination LOGIC itself is unit-tested
 * (lib/market-reading/__tests__/useCandles.test.ts — merge/floor/preserve — and
 * lib/chart/__tests__/placeLabels.test.ts — label de-collision); driving the
 * lightweight-charts canvas with synthetic wheel/drag from Playwright is
 * unreliable, so the interactive cascade is verified live rather than here.
 */

const VIEWPORTS = [
  { name: 'desktop-1280', width: 1280, height: 800 },
  { name: 'mobile-390', width: 390, height: 844 },
] as const;

// A large contiguous history whose last bar lines up with the reading's close,
// serving `before` (older page) + `has_more` exactly like the real endpoint.
const LAST = Math.floor(Date.UTC(2026, 4, 26, 11, 45) / 1000);
const FULL = Array.from({ length: 1000 }, (_, i) => {
  const close = 2300 + i * 0.5;
  return { time: LAST - (999 - i) * 900, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
});

function candlesRoute(route: Route) {
  const url = new URL(route.request().url());
  const limit = Number(url.searchParams.get('limit') ?? 400);
  const before = url.searchParams.get('before');
  let startI: number;
  let end: number;
  if (before != null) {
    const idx = FULL.findIndex((c) => c.time >= Number(before));
    end = idx < 0 ? FULL.length : idx;
    startI = Math.max(0, end - limit);
  } else {
    end = FULL.length;
    startI = Math.max(0, FULL.length - limit);
  }
  return route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ instrument: 'XAUUSD', timeframe: 'M15', candles: FULL.slice(startI, end), has_more: startI > 0 }),
  });
}

const READING = { ...FIXTURE_XAU_M15, header: { ...FIXTURE_XAU_M15.header, analysis_window_bars: 300 } };

async function gotoApp(page: Page, w: number, h: number) {
  await page.setViewportSize({ width: w, height: h });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.route('**/api/market-reading**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(READING) }),
  );
  await page.route('**/api/candles**', candlesRoute);
  await page.goto('/app?instrument=XAUUSD&timeframe=M15');
  await dismissCookieBanner(page);
  if (w < 1280) {
    const lecture = page.getByRole('tab', { name: /Lecture/i });
    await lecture.waitFor({ state: 'visible', timeout: 8_000 });
    await lecture.click();
  }
  await expect(page.getByRole('img', { name: /XAUUSD/i }).first()).toBeVisible({ timeout: 15_000 });
}

for (const vp of VIEWPORTS) {
  test.describe(`CHART-1 chart controls — ${vp.name}`, () => {
    test('renders with the full zoom + history control set', async ({ page }) => {
      await gotoApp(page, vp.width, vp.height);
      await expect(page.getByRole('button', { name: /Zoom avant|Zoom in/i })).toBeVisible();
      await expect(page.getByRole('button', { name: /Zoom arri|Zoom out/i })).toBeVisible();
      await expect(page.getByRole('button', { name: /Ajuster|Fit/i })).toBeVisible();
      // The "jump to latest" control only appears when history pagination is
      // wired — its presence confirms the chart received the history handles.
      await expect(
        page.getByRole('button', { name: /bougie la plus récente|most recent candle/i }),
      ).toBeVisible();
    });

    test('the controls are operable and never blank the chart', async ({ page }) => {
      await gotoApp(page, vp.width, vp.height);
      const chart = page.getByRole('img', { name: /XAUUSD/i }).first();
      for (const name of [/Zoom avant/i, /Zoom arri/i, /Ajuster/i, /bougie la plus récente/i]) {
        await page.getByRole('button', { name }).click();
        await page.waitForTimeout(150);
        await expect(chart).toBeVisible();
      }
    });
  });
}
