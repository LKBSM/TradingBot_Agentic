import { expect, test } from '@playwright/test';

/**
 * RG-1 — enriched « Régime de marché » panel. The tile grid + two-tab detail
 * panel (Donnée / Concept) and the reference-level tracing are exercised in full
 * by the component unit tests (components/app/__tests__/rg1-regime.test.tsx) and
 * a manual mock-data pass (documented in the audit). Those behaviours need a live
 * reading, which the e2e server does not provide — so here we assert the
 * structural guarantees that hold WITHOUT the data backend, and, where the card
 * DOES render, that a tile opens an in-card panel on « Donnée » and the « ? »
 * opens « Concept », one panel at a time, without changing the card's width.
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

test('1280×800: /app has no horizontal overflow and no raw regimePanel key', async ({ page }) => {
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

test('the Régime panel opens Donnée from a tile and Concept from « ? », keeping its width', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto('/app');
  const card = page.locator('.card', { hasText: 'Régime de marché' }).first();
  // The card needs a live reading; skip cleanly when the backend is absent.
  if ((await card.count()) === 0) {
    test.skip(true, 'no reading backend in this e2e environment');
    return;
  }

  const grid = card.locator('.tgrid');
  await expect(grid).toBeVisible();
  const widthBefore = (await card.boundingBox())?.width ?? 0;

  // A tile opens the detail panel on the « Donnée » tab.
  const tile = card.locator('.tile').first();
  await tile.click();
  const panel = card.locator('.tdetail');
  await expect(panel).toBeVisible();
  await expect(panel.locator('.ttab.on')).toHaveText(/Donnée/);

  // The tile's « ? » switches the same panel to « Concept » — with its mandatory
  // « ce que ça ne dit pas » block.
  await tile.getByRole('button', { name: /Expliquer cette mesure/i }).click();
  await expect(panel.locator('.ttab.on')).toHaveText(/Concept/);
  await expect(panel.locator('.notsay')).toBeVisible();

  // Only ONE detail panel is open in the whole card.
  await expect(card.locator('.tdetail')).toHaveCount(1);

  // The card kept its width (the panel scrolls internally, never full-width).
  const widthAfter = (await card.boundingBox())?.width ?? 0;
  expect(Math.abs(widthAfter - widthBefore)).toBeLessThanOrEqual(1);

  // No horizontal overflow introduced by the open panel.
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
});
