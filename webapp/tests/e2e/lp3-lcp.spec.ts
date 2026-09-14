import { test, type Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { dismissCookieBanner } from './utils';

/**
 * LP-3 — the LCP probe, and the before/after page shots.
 *
 * It deliberately assumes NOTHING the new hero introduces (no `[data-seq]`), so
 * the exact same spec can be pointed at either version of the landing. That is
 * how the "avant" figures in docs/audits/AUDIT-lp-3-plein-cadre.md were taken:
 *
 *   git checkout origin/main -- webapp/components/landing webapp/lib/landing webapp/messages
 *   npm run build
 *   CI=1 PORT=3141 E2E_BASE_URL=http://localhost:3141 LP3_TAG=avant  *     npx playwright test tests/e2e/lp3-lcp.spec.ts --project=chromium-desktop --workers=1
 *   git checkout HEAD -- webapp/components/landing webapp/lib/landing webapp/messages
 *
 * `LP3_TAG` names the run; the shots and the LCP reading land in
 * docs/audits/lp-3/ under that tag.
 */

const OUT = path.resolve(__dirname, '../../../docs/audits/lp-3');
const TAG = process.env.LP3_TAG ?? 'avant';
const VIEWPORTS = [
  { tag: '1280x800', width: 1280, height: 800 },
  { tag: '390x844', width: 390, height: 844 },
];

/** Largest Contentful Paint, read from the browser's own entries. */
async function lcp(page: Page): Promise<number> {
  return page.evaluate(
    () =>
      new Promise<number>((resolve) => {
        let last = 0;
        try {
          new PerformanceObserver((list) => {
            for (const e of list.getEntries()) last = e.startTime;
          }).observe({ type: 'largest-contentful-paint', buffered: true });
        } catch {
          resolve(-1);
          return;
        }
        setTimeout(() => resolve(Math.round(last)), 2_500);
      }),
  );
}

for (const vp of VIEWPORTS) {
  test(`${TAG} · ${vp.tag}`, async ({ page }) => {
    test.setTimeout(120_000);
    await page.setViewportSize({ width: vp.width, height: vp.height });
    await page.goto('/', { waitUntil: 'domcontentloaded', timeout: 60_000 });
    await dismissCookieBanner(page);
    const value = await lcp(page);
    await page.waitForTimeout(1_500);
    await page.screenshot({ path: path.join(OUT, `hero-${TAG}--${vp.tag}.png`) });
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(1_200);
    await page.screenshot({ path: path.join(OUT, `page-entiere-${TAG}--${vp.tag}.png`), fullPage: true });
    fs.mkdirSync(OUT, { recursive: true });
    fs.writeFileSync(
      path.join(OUT, `lcp-${TAG}-${vp.tag}.json`),
      JSON.stringify({ tag: TAG, viewport: vp.tag, lcpMs: value }, null, 2),
    );
    // eslint-disable-next-line no-console
    console.log(`LCP ${TAG} ${vp.tag} = ${value} ms`);
  });
}
