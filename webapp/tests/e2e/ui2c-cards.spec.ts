import { expect, test } from '@playwright/test';

/**
 * UI-2c — Structure / Liquidité / aide. The interactive card behaviours (sort
 * panel toggle, help one-at-a-time, row→chart highlight, "En savoir plus" →
 * /zones?zone=<id>, filter→empty, tick-without-reorder) are covered by the
 * component unit tests in components/app/__tests__/ui2c.test.tsx and by a
 * manual mock-data e2e pass (documented in the audit). These cards render only
 * with a live reading, which the e2e server does not provide — so here we assert
 * the structural guarantees that hold WITHOUT the data backend: at 1280 the App
 * has no horizontal overflow and leaks no raw i18n key; and where the cards DO
 * render (data present), the "?" help opens an in-card panel.
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

test('1280×800: /app has no horizontal overflow and no raw i18n key', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/app');
  await expect(
    page.getByRole('complementary', { name: /combinaisons disponibles/i }),
  ).toBeVisible();

  const raw = await page.evaluate(RAW_KEY_SCAN);
  expect(raw).toEqual([]);

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
});

test('the Structure card help toggles an in-card panel when the card is present', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto('/app');
  const card = page.locator('.card', { hasText: 'Structure' }).first();
  // Skip cleanly when the reading backend isn't available in this environment
  // (the cards need a live reading — their interactions are unit-tested).
  if ((await card.count()) === 0) {
    test.skip(true, 'no reading backend in this e2e environment');
    return;
  }
  const help = card.getByRole('button', { name: /En savoir plus/i });
  await help.click();
  await expect(card.locator('.infobox.on')).toBeVisible();
  await help.click();
  await expect(card.locator('.infobox.on')).toHaveCount(0);
});
