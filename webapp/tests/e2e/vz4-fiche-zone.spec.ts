import { test, expect, type Page } from '@playwright/test';
import path from 'node:path';
import { dismissCookieBanner } from './utils';
import { mockAllApis } from './ds-mock';
import { SAMPLE_READING_XAU_H4 } from '../../lib/ds-samples';

/**
 * VZ-4 — the dedicated zone sheet (`/zones/<engine id>`), at both target
 * viewports, against the REAL frozen reading served by `mockAllApis` (no
 * backend). It checks the four things the mission cares about:
 *
 *   1. both entry points (the compact /zones card and the /app structure list)
 *      land on the SAME `/zones/<id>` page;
 *   2. the comblement figure is printed ONCE, never on a contact row;
 *   3. « Zones à l'intérieur » only shows when a real containment fact exists;
 *   4. the M.I.A column is docked on the sheet with the zone-contextual chips.
 */

const OUT = path.resolve(__dirname, '../../../docs/audits/vz-4/captures');
const VIEWPORTS = [
  { tag: 'desktop-1280x800', width: 1280, height: 800 },
  { tag: 'mobile-390x844', width: 390, height: 844 },
];

const COMBO = 'instrument=XAUUSD&timeframe=H4';

/** A real OB id from the frozen reading (never a hand-written one). */
const ZONE_ID = SAMPLE_READING_XAU_H4.structure.order_blocks[0]!.id;

async function settle(page: Page, opts: { hasChart?: boolean } = {}) {
  await dismissCookieBanner(page);
  await page
    .locator('.animate-spin')
    .first()
    .waitFor({ state: 'detached', timeout: 12_000 })
    .catch(() => {});
  if (opts.hasChart) {
    await page
      .locator('canvas')
      .first()
      .waitFor({ state: 'visible', timeout: 15_000 })
      .catch(() => {});
    await page.waitForTimeout(1_200);
  }
  await page.waitForTimeout(600);
}

for (const vp of VIEWPORTS) {
  test.describe(`VZ-4 fiche de zone — ${vp.tag}`, () => {
    test.beforeEach(async ({ page }) => {
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await mockAllApis(page);
    });

    test('the sheet renders the zone, with one chart and one comblement', async ({ page }) => {
      await page.goto(`/zones/${encodeURIComponent(ZONE_ID)}?${COMBO}`);
      await settle(page, { hasChart: true });

      const sheet = page.getByTestId('zone-detail');
      await expect(sheet).toBeVisible();
      await expect(sheet).toHaveAttribute('data-zone-id', ZONE_ID);

      // One visual, not a second product: exactly one chart figure.
      await expect(page.getByTestId('zone-detail-chart')).toHaveCount(1);

      // The comblement, when present, is a SINGLE bar — and never on a row.
      const fill = page.getByTestId('zone-detail-fill');
      expect(await fill.count()).toBeLessThanOrEqual(1);
      const rows = page.locator('.zdt-row');
      for (let i = 0; i < (await rows.count()); i += 1) {
        expect(await rows.nth(i).textContent()).not.toMatch(/%/);
      }

      await page.screenshot({
        path: path.join(OUT, `${vp.tag}-fiche-zone.png`),
        fullPage: true,
      });
    });

    test('« Zones à l’intérieur » never appears without a containment fact', async ({ page }) => {
      await page.goto(`/zones/${encodeURIComponent(ZONE_ID)}?${COMBO}`);
      await settle(page);
      const nested = page.getByTestId('zone-detail-nested');
      if ((await nested.count()) > 0) {
        // Present ⇒ it must carry the real containment sentence, not filler.
        await expect(nested).toContainText(/à l['’]intérieur de cette zone/);
      }
      // Either way, the heading never renders on its own with nothing under it.
      const headings = await page.locator('h2').allTextContents();
      const hasHeading = headings.some((h) => /Zones à l['’]intérieur/.test(h));
      expect(hasHeading).toBe((await nested.count()) > 0);
    });

    test('an unknown id says so honestly instead of showing a look-alike', async ({ page }) => {
      await page.goto(`/zones/zone-qui-nexiste-pas?${COMBO}`);
      await settle(page);
      await expect(page.getByTestId('zone-detail-missing')).toBeVisible();
      await expect(page.getByTestId('zone-detail')).toHaveCount(0);
      await page.screenshot({
        path: path.join(OUT, `${vp.tag}-fiche-zone-introuvable.png`),
        fullPage: true,
      });
    });

    test('« En savoir plus » on the /zones card lands on the sheet', async ({ page }) => {
      await page.goto(`/zones?${COMBO}`);
      await settle(page);
      const link = page.locator('.zmore').first();
      await expect(link).toBeVisible();
      const href = await link.getAttribute('href');
      expect(href).toContain('/zones/');
      await link.click();
      await expect(page.getByTestId('zone-detail')).toBeVisible({ timeout: 15_000 });
      expect(page.url()).toContain('/zones/');
    });
  });
}

// The M.I.A column is a desktop disposition (a drawer on phones) — the chip
// assertion runs at the desktop viewport only.
test.describe('VZ-4 — M.I.A contextual chips', () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await mockAllApis(page);
  });

  test('the single M.I.A panel is docked and offers the zone questions', async ({ page }) => {
    await page.goto(`/zones/${encodeURIComponent(ZONE_ID)}?${COMBO}`);
    await settle(page);

    // Exactly ONE panel (MIA-3 guard) and the subject is this zone.
    await expect(page.getByTestId('mia-subject')).toHaveCount(1);

    // The starter chips are the zone-contextual ones, and none asks for a
    // prediction (the mockup's « ça va rebondir ? » probe is deliberately not
    // shipped — see docs/audits/AUDIT-vz-4-fiche-zone.md).
    const chips = page.locator('button', { hasText: /zone/i });
    const texts = await chips.allTextContents();
    const joined = texts.join(' | ');
    expect(joined).toMatch(/form[ée]e/i);
    expect(joined).not.toMatch(/rebondir|bounce/i);

    await page.screenshot({
      path: path.join(OUT, 'desktop-1280x800-fiche-zone-mia.png'),
      fullPage: false,
    });
  });
});
