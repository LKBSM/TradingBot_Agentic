import { test, expect, type Page } from '@playwright/test';
import path from 'node:path';
import { dismissCookieBanner } from './utils';

/**
 * LP-3 « plein cadre » — the visual record and the two behavioural promises.
 *
 * The captures go to docs/audits/lp-3/ and are what the founder confirms live
 * before the merge: the hero mid-arrival and settled, every section, and the
 * hero on the four themes, at both resolutions.
 *
 * Two tests here are not captures but contracts, because both are invisible
 * when they break:
 *   · with `prefers-reduced-motion: reduce`, the hero is rendered FINISHED and
 *     nothing animates;
 *   · a click on a layer DURING the arrival stops it and is applied.
 */

const OUT = path.resolve(__dirname, '../../../docs/audits/lp-3');
const THEMES = ['terminal', 'atelier', 'schema', 'ardoise'] as const;
const VIEWPORTS = [
  { tag: '1280x800', width: 1280, height: 800 },
  { tag: '390x844', width: 390, height: 844 },
];

/** The whole arrival, plus a margin: candles → layers → reading → refusal. */
const SEQUENCE_MS = 7_200;

async function open(page: Page, opts: { theme?: string; reduce?: boolean } = {}) {
  if (opts.theme) {
    await page.addInitScript((t) => {
      try { localStorage.setItem('theme', t as string); } catch { /* private mode */ }
    }, opts.theme);
  }
  if (opts.reduce) await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/', { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await dismissCookieBanner(page);
}

const stage = (page: Page) => page.locator('[data-seq]');

for (const vp of VIEWPORTS) {
  test.describe(`LP-3 · ${vp.tag}`, () => {
    test.use({ viewport: { width: vp.width, height: vp.height } });
    test.describe.configure({ retries: 2 });

    test('hero: mid-arrival then settled', async ({ page }) => {
      test.setTimeout(90_000);
      await open(page);
      // MID-ARRIVAL — caught while the window is still drawing itself.
      await expect(stage(page)).toHaveAttribute('data-seq', /idle|typing/);
      await page.waitForTimeout(1_400);
      await page.screenshot({ path: path.join(OUT, `hero-pendant--${vp.tag}.png`) });
      // SETTLED — the sequence has run its course on its own.
      await expect(stage(page)).toHaveAttribute('data-seq', 'done', { timeout: SEQUENCE_MS });
      await page.waitForTimeout(900);
      await page.screenshot({ path: path.join(OUT, `hero-apres--${vp.tag}.png`) });
      // Desktop: the refusal is the point of the sequence — decision (e) shortened
      // the chart precisely so it lands ABOVE the fold on a 800px-tall laptop.
      // Mobile stacks the hero (text, then the window): the window is below the
      // fold by construction there, so the promise is that it is reachable and
      // settled, not that it is on the first screen.
      const refusal = page.getByText(/Je ne réponds pas à ça/).first();
      if (vp.width >= 1080) {
        await expect(refusal).toBeInViewport();
      } else {
        await refusal.scrollIntoViewIfNeeded();
        await expect(refusal).toBeVisible();
      }
    });

    test('the whole page, section by section', async ({ page }) => {
      test.setTimeout(120_000);
      await open(page);
      await page.waitForTimeout(SEQUENCE_MS);
      // Walk the page so every Reveal has fired before the full-page shot.
      for (const id of ['mia', 'demo', 'outils', 'comment', 'honnetete', 'tarifs', 'faq']) {
        await page.locator(`#${id}`).scrollIntoViewIfNeeded();
        await page.waitForTimeout(700);
        await page.locator(`#${id}`).screenshot({ path: path.join(OUT, `section-${id}--${vp.tag}.png`) });
      }
      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      await page.waitForTimeout(900);
      await page.screenshot({ path: path.join(OUT, `page-entiere--${vp.tag}.png`), fullPage: true });
    });
  });
}

test.describe('LP-3 · hero on the four themes', () => {
  test.describe.configure({ retries: 2 });
  for (const theme of THEMES) {
    for (const vp of VIEWPORTS) {
      test(`${theme} · ${vp.tag}`, async ({ page }) => {
        test.setTimeout(90_000);
        await page.setViewportSize({ width: vp.width, height: vp.height });
        await open(page, { theme });
        await expect(page.locator('html')).toHaveAttribute('data-design', theme);
        await expect(stage(page)).toHaveAttribute('data-seq', 'done', { timeout: SEQUENCE_MS });
        await page.waitForTimeout(700);
        await page.screenshot({ path: path.join(OUT, `hero-theme-${theme}--${vp.tag}.png`) });
      });
    }
  }
});

test.describe('LP-3 · the two promises of the arrival', () => {
  test.describe.configure({ retries: 2 });
  test.use({ viewport: { width: 1280, height: 800 } });

  test('reduced motion: the hero is rendered finished, and nothing animates', async ({ page }) => {
    await open(page, { reduce: true });
    // Settled on arrival — not after the sequence, which never starts.
    await expect(stage(page)).toHaveAttribute('data-seq', 'done');
    // The finished window: the whole reading and the closing refusal.
    await expect(page.getByText(/CHOCH haussier/).first()).toBeVisible();
    await expect(page.getByText(/liquidité achat reste intacte/).first()).toBeVisible();
    await expect(page.getByText(/Je ne réponds pas à ça/).first()).toBeVisible();
    // And nothing is running: every animated element sits at its end state.
    const animating = await page.evaluate(() => {
      const root = document.querySelector('[data-seq]');
      if (!root) return -1;
      return root.getAnimations({ subtree: true }).filter((a) => a.playState === 'running').length;
    });
    expect(animating).toBe(0);
    await page.screenshot({ path: path.join(OUT, 'hero-reduced-motion--1280x800.png') });
  });

  test('a click on a layer during the arrival stops it AND is applied', async ({ page }) => {
    await open(page);
    // We are mid-arrival on purpose.
    await expect(stage(page)).toHaveAttribute('data-seq', /idle|typing/);
    const chip = stage(page).getByRole('button', { name: 'Fair Value Gaps' });
    await chip.click();
    // 1. the sequence yielded, immediately
    await expect(stage(page)).toHaveAttribute('data-seq', 'done');
    // 2. the click did what it says: the FVG fragment left the reading…
    await expect(stage(page).getByText(/Fair Value Gap baissier comblé/)).toHaveCount(0);
    // …and what the visitor kept is still described
    await expect(stage(page).getByText(/Order Block haussier/).first()).toBeVisible();
    await expect(chip).toHaveAttribute('aria-pressed', 'false');
    await page.screenshot({ path: path.join(OUT, 'hero-interrompu--1280x800.png') });
  });
});
