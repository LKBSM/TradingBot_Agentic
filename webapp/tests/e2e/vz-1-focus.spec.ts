import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * VZ-1 — the unified click→select gesture. Observable at the DOM level via
 * `aria-pressed` on the panel rows, because the selection is a SINGLE piece of
 * provider state shared product-wide (StructureCard renders under the shared
 * ChartViewProvider).
 *
 * DETTE-1 repoint: this previously ran against the landing multi-market gallery,
 * which LP-1's home-page refonte removed. It now runs on /app with the
 * market-reading + candles endpoints mocked per the PERF-1 pattern
 * (perf1-load-honesty.spec.ts), so it stays backend-free and deterministic.
 * The camera framing math itself is covered by focusController unit tests.
 */

function makeCandles(n = 150) {
  const base = 2300;
  const start = Math.floor(Date.UTC(2026, 5, 20) / 1000);
  const candles = Array.from({ length: n }, (_, i) => {
    const close = base + i * 2;
    return { time: start + i * 900, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
  });
  return { instrument: 'XAUUSD', timeframe: 'M15', candles };
}

async function mockReading(page: Page) {
  await page.route('**/api/candles**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(makeCandles()) }),
  );
  await page.route('**/api/market-reading**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FIXTURE_XAU_M15) }),
  );
}

/**
 * Loads /app with the reading mocked and returns the first selectable zone row.
 * /app has TWO structure UIs sharing the ChartViewProvider: desktop (≥1280)
 * renders StructureCard (`.zrow`); the stacked mobile layout (<1280) renders
 * ReadingColumn → StructureSection (an accordion with "· actif" rows). The VZ-1
 * gesture is identical on both — only the DOM differs.
 */
async function openFirstStructure(page: Page): Promise<ReturnType<Page['locator']>> {
  // /app first-compile under `next dev` (esp. the stacked mobile layout) can
  // exceed the 30s default; the gesture itself is fast once loaded.
  test.setTimeout(90_000);
  await mockReading(page);
  await page.goto('/app?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
  await dismissCookieBanner(page);
  const width = page.viewportSize()?.width ?? 1280;
  if (width < 1280) {
    const lecture = page.getByRole('tab', { name: /Lecture/i });
    await lecture.waitFor({ state: 'visible', timeout: 60_000 });
    await lecture.click();
    // Reveal the StructureSection accordion, then take its first active zone row.
    const trigger = page.getByRole('button', { name: /Structure de marché/i }).first();
    await trigger.waitFor({ state: 'visible', timeout: 60_000 });
    await trigger.scrollIntoViewIfNeeded();
    if ((await trigger.getAttribute('aria-expanded')) === 'false') await trigger.click();
    const zone = page.getByRole('button', { name: /· actif|active/i }).first();
    await zone.waitFor({ state: 'visible', timeout: 60_000 });
    return zone;
  }
  const zone = page.locator('.zrow').first();
  await zone.waitFor({ state: 'visible', timeout: 60_000 });
  return zone;
}

test.describe('VZ-1 unified selection @ 1280×800', () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test('clicking a zone selects it; re-clicking toggles it off', async ({ page }) => {
    const zone = await openFirstStructure(page);
    await expect(zone).toHaveAttribute('aria-pressed', 'false');
    await zone.click();
    await expect(zone).toHaveAttribute('aria-pressed', 'true');
    await zone.click();
    await expect(zone).toHaveAttribute('aria-pressed', 'false');
  });

  test('Escape deselects the active element', async ({ page }) => {
    const zone = await openFirstStructure(page);
    await zone.click();
    await expect(zone).toHaveAttribute('aria-pressed', 'true');
    await page.keyboard.press('Escape');
    await expect(zone).toHaveAttribute('aria-pressed', 'false');
  });

  test('single selection across families: a BOS event deselects the active zone', async ({ page }) => {
    const zone = await openFirstStructure(page);
    // The BOS/CHOCH event rows share the `.strow` class; pick the BOS one.
    const bos = page.locator('.strow').filter({ hasText: 'BOS' }).first();
    await zone.click();
    await expect(zone).toHaveAttribute('aria-pressed', 'true');
    await bos.click();
    await expect(bos).toHaveAttribute('aria-pressed', 'true');
    // Only one element is selected product-wide — the zone let go.
    await expect(zone).toHaveAttribute('aria-pressed', 'false');
  });
});

test.describe('VZ-1 gesture on a narrow window @ 390×844', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('the same click→select gesture works under 768px', async ({ page }) => {
    const zone = await openFirstStructure(page);
    await zone.click();
    await expect(zone).toHaveAttribute('aria-pressed', 'true');
    await page.keyboard.press('Escape');
    await expect(zone).toHaveAttribute('aria-pressed', 'false');
  });
});
