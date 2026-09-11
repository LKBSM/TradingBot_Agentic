import { test, type Page } from '@playwright/test';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { dismissCookieBanner } from './utils';

/**
 * LP-3 — before/after shots of the home page, at the two reference viewports.
 *
 * Not a guard (home.test.tsx + lp1-accueil.spec.ts hold the assertions) — this
 * exists so the founder can compare the redesign against what it replaced, and
 * so the shots can be regenerated from a single command:
 *
 *   LP3_PHASE=before PORT=3193 CI=1 E2E_BASE_URL=http://localhost:3193 \
 *     npx playwright test tests/e2e/lp3-shots.spec.ts --workers=1
 *
 * Output: docs/audits/lp-3/shots/{before,after}/ at the REPO root, never
 * webapp/docs (a lesson from STR-2).
 */
const PHASE = process.env.LP3_PHASE === 'after' ? 'after' : 'before';
const SHOTS = path.resolve(__dirname, '../../../docs/audits/lp-3/shots', PHASE);

const CASES = [
  { code: 'fr', path: '/' },
  { code: 'en', path: '/en' },
];

const VIEWPORTS = [
  { name: 'desktop-1280x800', width: 1280, height: 800 },
  { name: 'mobile-390x844', width: 390, height: 844 },
];

async function open(page: Page, url: string) {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 90_000 });
  await dismissCookieBanner(page);
  const reject = page.getByRole('button', { name: /Tout refuser|Reject all/i });
  if (await reject.count()) await reject.first().click({ timeout: 3_000 });
  // Freeze every animation so two runs of the same page are pixel-comparable.
  await page.addStyleTag({
    content:
      '*,*::before,*::after{animation:none!important;transition:none!important}html{scroll-behavior:auto!important}',
  });
  await page.waitForTimeout(600);
}

test.describe(`LP-3 — home shots (${PHASE})`, () => {
  test.beforeAll(() => mkdirSync(SHOTS, { recursive: true }));

  for (const c of CASES) {
    for (const vp of VIEWPORTS) {
      test(`${c.code} · ${vp.name}`, async ({ page }) => {
        await page.setViewportSize({ width: vp.width, height: vp.height });
        await open(page, c.path);

        // The fold — what a visitor actually sees before touching anything.
        await page.screenshot({
          path: path.join(SHOTS, `fold-${c.code}-${vp.name}.png`),
        });

        // The whole page, for the section-by-section comparison.
        await page.screenshot({
          path: path.join(SHOTS, `full-${c.code}-${vp.name}.png`),
          fullPage: true,
        });
      });
    }
  }
});

/**
 * LP-3 — the choreography itself.
 *
 * The shots above run with reduced motion, which is the DEGRADED path (the
 * whole reading at once). These three capture the enhanced path at its three
 * meaningful positions, so the founder can see what a visitor who scrolls
 * actually gets without running the app.
 */
test.describe('LP-3 — the pinned scroll section', () => {
  test.use({ viewport: { width: 1280, height: 900 } });

  const STEPS = ['La cassure', 'La zone', 'La liquidité'];

  test('three positions of the reading assembling itself', async ({ page }) => {
    mkdirSync(SHOTS, { recursive: true });
    await page.goto('/', { waitUntil: 'domcontentloaded', timeout: 90_000 });
    await dismissCookieBanner(page);
    await page.waitForTimeout(400);

    const section = page.locator('#lecture');
    for (const [i, label] of STEPS.entries()) {
      await section.getByText(label, { exact: true }).first().scrollIntoViewIfNeeded();
      await page.waitForTimeout(700);
      await page.screenshot({ path: path.join(SHOTS, `unfold-${i + 1}-${label.replace(/\s/g, '-')}.png`) });
    }
  });
});
