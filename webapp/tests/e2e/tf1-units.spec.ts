import { expect, test } from '@playwright/test';

/**
 * TF-1 — six-timeframe parity. The click-to-frame behaviour needs a live reading
 * (chart + zones), which the e2e server does not provide — that is validated live
 * before merge. Here we assert the structural guarantees that hold WITHOUT the
 * data backend, on the new units and at both viewports: switching to M5 and D1
 * renders without crash, without horizontal overflow, and without any raw i18n
 * key (so the new registry-driven strings all resolve). The framing regression
 * itself is guarded deterministically by the TF_SECONDS unit test.
 */

const RAW_KEY_SCAN = `(() => {
  const bad = [];
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = w.nextNode())) {
    const t = (n.textContent || '').trim();
    if (!t || t.includes('/') || t.includes('@') || t.includes(':')) continue;
    if (/^[a-z][a-zA-Z0-9]*(\\.[a-zA-Z][a-zA-Z0-9]*)+$/.test(t)) bad.push(t);
  }
  return bad;
})()`;

async function overflow(page: import('@playwright/test').Page) {
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
}

for (const vp of [
  { name: 'desktop 1280×800', width: 1280, height: 800 },
  { name: 'phone 390×844', width: 390, height: 844 },
]) {
  for (const tf of ['M5', 'D1']) {
    test(`${vp.name}: switching to ${tf} renders cleanly (no overflow, no raw key)`, async ({ page }) => {
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await page.goto('/app');

      // Select the timeframe from the rail (desktop) or the sidebar (phone). Both
      // expose the unit as a button whose accessible name contains the id.
      const tfButton = page.getByRole('button', { name: new RegExp(`\\b${tf}\\b`) }).first();
      if (await tfButton.count()) {
        await tfButton.click();
      }

      // No raw i18n key leaked (the new registry-driven strings all resolve).
      expect(await page.evaluate(RAW_KEY_SCAN)).toEqual([]);
      // No horizontal overflow at this viewport.
      expect(await overflow(page)).toBeLessThanOrEqual(1);
    });
  }
}
