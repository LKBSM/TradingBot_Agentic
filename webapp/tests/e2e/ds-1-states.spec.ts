import { test, expect, type Page } from '@playwright/test';
import path from 'node:path';
import { dismissCookieBanner } from './utils';
import { mockAllApis } from './ds-mock';
import { SAMPLE_SCAN_CONFIG } from '../../lib/ds-samples';

/**
 * DS-1 — curated KEY STATES that the routes×themes matrix (ds-1-coverage) does
 * not reach because they need a seeded local state or a tab switch. These are
 * distinct SURFACES of the current UI, not transient loading states:
 *   · scanner RESULTS (the matrix shows the empty builder) — seed a saved config
 *   · mobile /app « Lecture » (chart) and « Chat » tabs (default tab is Marchés)
 * Captured in the two most-used themes (terminal dark default, atelier light).
 * The harness can produce any other theme/viewport on demand.
 */

const OUT = path.resolve(__dirname, '../../../docs/audits/ds-1/coverage');
const THEMES = ['terminal', 'atelier'] as const;

async function withTheme(page: Page, theme: string) {
  await page.addInitScript((t) => { try { localStorage.setItem('theme', t as string); } catch {} }, theme);
}
async function seedScannerConfig(page: Page) {
  const cfg = JSON.stringify(SAMPLE_SCAN_CONFIG);
  await page.addInitScript((c) => { try { localStorage.setItem('mia.conditionsConfig.v1', c as string); } catch {} }, cfg);
}

for (const theme of THEMES) {
  test(`scanner-results · ${theme} · desktop`, async ({ page }) => {
    test.setTimeout(90_000);
    await withTheme(page, theme);
    await seedScannerConfig(page);
    await mockAllApis(page);
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('/scanner', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);
    await page.locator('.combo').first().waitFor({ state: 'visible', timeout: 15_000 }).catch(() => {});
    await page.waitForTimeout(800);
    await expect(page.locator('.combo').first()).toBeVisible();
    await page.screenshot({ path: path.join(OUT, `scanner-results--${theme}--desktop.png`), fullPage: true });
  });

  test(`app-lecture · ${theme} · mobile`, async ({ page }) => {
    test.setTimeout(90_000);
    await withTheme(page, theme);
    await mockAllApis(page);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/app?instrument=XAUUSD&timeframe=H4', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);
    await page.getByRole('tab', { name: 'Lecture' }).click({ timeout: 8_000 }).catch(() => {});
    await page.locator('canvas').first().waitFor({ state: 'visible', timeout: 15_000 }).catch(() => {});
    await page.waitForTimeout(2_000);
    await page.screenshot({ path: path.join(OUT, `app-lecture--${theme}--mobile.png`), fullPage: true });
  });

  test(`app-chat · ${theme} · mobile`, async ({ page }) => {
    test.setTimeout(90_000);
    await withTheme(page, theme);
    await mockAllApis(page);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/app?instrument=XAUUSD&timeframe=H4', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);
    await page.getByRole('tab', { name: 'Chat' }).click({ timeout: 8_000 }).catch(() => {});
    await page.waitForTimeout(1_500);
    await page.screenshot({ path: path.join(OUT, `app-chat--${theme}--mobile.png`), fullPage: true });
  });
}
