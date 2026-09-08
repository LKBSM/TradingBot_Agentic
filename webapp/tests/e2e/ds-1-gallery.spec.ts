import { test, expect } from '@playwright/test';
import path from 'node:path';

/**
 * DS-1 — the design gallery renders with NO backend running (only `next dev`,
 * no FastAPI). Every surface is present, fed by the frozen real-data samples.
 * Captured at both target viewports for Claude Design.
 *
 * The gallery is a dev-only route (production 404s), so this runs against the
 * dev server (E2E_BASE_URL) — never a production build.
 */

const OUT = path.resolve(__dirname, '../../../docs/audits/ds-1');
const SURFACES = ['market-reading', 'zones', 'chart', 'scanner', 'mia', 'landing'];

async function assertGallery(page: import('@playwright/test').Page) {
  await page.goto('/galerie', { waitUntil: 'networkidle' });
  await expect(page.getByTestId('ds-gallery')).toBeVisible();
  for (const s of SURFACES) {
    await expect(page.getByTestId(`ds-surface-${s}`)).toBeVisible();
  }
  // the price-inside-band zone (the gauge edge case) is present
  await expect(page.getByTestId('ds-state-prix-à-l’intérieur')).toBeVisible();
}

test('gallery renders without a backend — desktop 1280×800', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await assertGallery(page);
  // give the canvas chart a beat to paint before the capture
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(OUT, 'gallery-desktop-1280x800.png'), fullPage: true });
});

test('gallery renders without a backend — mobile 390×844', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await assertGallery(page);
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(OUT, 'gallery-mobile-390x844.png'), fullPage: true });
});
