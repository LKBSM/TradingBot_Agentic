import { test, expect, type Page } from '@playwright/test';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { dismissCookieBanner } from './utils';

/**
 * LP-2S — visual proof that the four-stat banner is gone and that the line that
 * replaced it renders correctly at both reference viewports.
 *
 * Not a guard (home.test.tsx and lp1-accueil.spec.ts hold the assertions) — this
 * exists so the founder can confirm the rendering without running the app, and
 * so the shots can be regenerated after any hero change:
 *
 *   PORT=3188 CI=1 E2E_BASE_URL=http://localhost:3188 \
 *     npx playwright test tests/e2e/lp2s-shots.spec.ts --workers=1 --project=chromium-desktop
 *
 * Output: docs/audits/lp-2s/shots/ at the REPO root (not webapp/docs).
 */
const SHOTS = path.resolve(__dirname, '../../../docs/audits/lp-2s/shots');

const CASES = [
  { code: 'fr', path: '/', line: /80 marchés au programme/i },
  { code: 'en', path: '/en', line: /80 markets planned/i },
];

const VIEWPORTS = [
  { name: 'desktop-1280x800', width: 1280, height: 800 },
  { name: 'mobile-390x844', width: 390, height: 844 },
];

async function open(page: Page, url: string) {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  // At 390×844 the consent banner covers the bottom of the hero — exactly where
  // the line sits. `toBeVisible()` would still pass (it does not test occlusion),
  // so the shot, not the assertion, is what catches this. The shared helper only
  // knows the fr label, so handle /en here rather than widen a helper every other
  // spec depends on.
  await dismissCookieBanner(page);
  const reject = page.getByRole('button', { name: /Tout refuser|Reject all/i });
  if (await reject.count()) await reject.first().click({ timeout: 3_000 });
  await expect(reject).toHaveCount(0);
  await page.addStyleTag({
    content: '*,*::before,*::after{animation:none!important;transition:none!important}html{scroll-behavior:auto!important}',
  });
}

test.describe('LP-2S — hero shots', () => {
  test.beforeAll(() => mkdirSync(SHOTS, { recursive: true }));

  for (const c of CASES) {
    for (const vp of VIEWPORTS) {
      test(`${c.code} · ${vp.name}`, async ({ page }) => {
        await page.setViewportSize({ width: vp.width, height: vp.height });
        await open(page, c.path);

        const line = page.getByText(c.line).first();
        await line.scrollIntoViewIfNeeded();
        await expect(line).toBeVisible();

        // the retired tiles must be nowhere on the page
        await expect(page.locator('body')).not.toContainText('structures détectées');
        await expect(page.locator('body')).not.toContainText('structures detected');

        await page.screenshot({
          path: path.join(SHOTS, `hero-${c.code}-${vp.name}.png`),
        });
        await line.screenshot({
          path: path.join(SHOTS, `line-${c.code}-${vp.name}.png`),
        });
      });
    }
  }
});
