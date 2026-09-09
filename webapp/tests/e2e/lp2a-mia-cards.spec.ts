import { test, expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

/**
 * LP-2A — the four M.I.A capability cards on the home page.
 *
 * Two jobs:
 *  1. assert the cards stay balanced after the copy cut — on one grid row the
 *     four cards must keep the SAME height (CSS grid `stretch`), so trimming
 *     three of them to one sentence cannot leave a ragged row;
 *  2. write before/after captures of the section at 1280×800 and 390×844 for
 *     the founder's live review (SHOT_DIR=before on the pre-LP-2A tree).
 */

const DIR = process.env.SHOT_DIR ?? 'after';
const OUT = path.join('..', 'docs', 'audits', 'lp2a-shots', DIR);

async function mock(page: Page) {
  await page.route('**/api/access/me', (r) =>
    r.fulfill({
      json: {
        authenticated: true, gate_enforced: false, beta_lockdown: false,
        must_login: false, is_owner: true, has_access: true, subscription_required: false,
      },
    }),
  );
  // Pre-decide the cookie consent: the banner is anchored bottom-right and
  // would otherwise cover the chat demo in every capture.
  await page.addInitScript(() => {
    window.localStorage.setItem(
      'mia.cookie-consent.v1',
      JSON.stringify({
        necessary: true, functional: false, analytics: false, marketing: false,
        decidedAt: '2026-01-01T00:00:00.000Z',
      }),
    );
  });
}

const VIEWPORTS = [
  { name: 'desktop', width: 1280, height: 800 },
  { name: 'mobile', width: 390, height: 844 },
] as const;

for (const vp of VIEWPORTS) {
  test(`shot ${DIR} fr mia-section ${vp.name}`, async ({ page }) => {
    fs.mkdirSync(OUT, { recursive: true });
    await mock(page);
    await page.setViewportSize({ width: vp.width, height: vp.height });
    await page.goto('/fr', { waitUntil: 'networkidle' });
    const section = page.locator('section#mia');
    await section.scrollIntoViewIfNeeded();
    await page.waitForTimeout(700);
    await section.screenshot({ path: path.join(OUT, `fr-mia-${vp.name}.png`) });
  });
}

test('the four cards keep an equal height on one desktop row', async ({ page }) => {
  await mock(page);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/fr', { waitUntil: 'networkidle' });
  const section = page.locator('section#mia');
  await section.scrollIntoViewIfNeeded();
  // The capability grid is the last direct grid of the section: 4 cards, 1 row.
  const cards = section.locator('h4').locator('..');
  await expect(cards).toHaveCount(4);
  const boxes = await cards.evaluateAll((els) =>
    els.map((el) => {
      const r = el.getBoundingClientRect();
      return { top: Math.round(r.top), height: Math.round(r.height) };
    }),
  );
  const heights = boxes.map((b) => b.height);
  const tops = boxes.map((b) => b.top);
  // Same row…
  expect(Math.max(...tops) - Math.min(...tops), `tops: ${tops.join(', ')}`).toBeLessThanOrEqual(1);
  // …and the same height, to the pixel (grid items stretch).
  expect(Math.max(...heights) - Math.min(...heights), `heights: ${heights.join(', ')}`).toBeLessThanOrEqual(1);
});

test('each card renders exactly one sentence of body copy', async ({ page }) => {
  await mock(page);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/fr', { waitUntil: 'networkidle' });
  const section = page.locator('section#mia');
  const bodies = section.locator('h4 + p');
  await expect(bodies).toHaveCount(4);
  for (const text of await bodies.allInnerTexts()) {
    // A sentence-final period is one followed by a space + capital, or the end
    // of the string. « 4 h » and abbreviations therefore never count.
    const ends = text.match(/\.(\s+[A-ZÀ-Þ«]|\s*$)/g) ?? [];
    expect(ends.length, `one sentence expected: "${text}"`).toBe(1);
  }
});
