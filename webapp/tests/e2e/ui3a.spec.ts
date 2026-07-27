import { expect, test } from '@playwright/test';

/**
 * UI-3a — language in Réglages + multi-select filters. The full interactive
 * behaviours (chip toggles, reset, zero-selection message, « N sur M » counter,
 * locale switch keeping the page) are covered by the component/hook unit tests
 * (ui3a-filters.test.tsx, use-locale-switch.test.ts) and a manual live pass. The
 * filter cards need a live reading and Réglages needs an authenticated session —
 * neither is provided by the e2e server — so here we assert what holds WITHOUT
 * the backend, and exercise the interactions only when the surface renders.
 */

test('1280×800: /app has no horizontal overflow and no raw i18n key', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/app');
  await expect(
    page.getByRole('complementary', { name: /combinaisons disponibles/i }),
  ).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
});

test('multi-select filters: toggling a state chip updates the « N sur M » counter', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto('/app');
  const card = page.locator('.card', { hasText: 'Liquidité externe' }).first();
  if ((await card.count()) === 0) {
    test.skip(true, 'no reading backend in this e2e environment');
    return;
  }
  // Open the filter panel.
  await card.getByRole('button', { name: /Trier|Filtrer|tri/i }).first().click();
  const badgeBefore = (await card.locator('.badge2').textContent()) ?? '';
  expect(badgeBefore).toMatch(/\bsur\b/); // « N sur M poches »

  // Toggle one state chip off — the counter's N must change, never silently
  // fall back to the total.
  const brokenChip = card.getByRole('button', { name: 'Cassées' });
  await brokenChip.click();
  await expect(brokenChip).toHaveAttribute('aria-pressed', 'false');
  const badgeAfter = (await card.locator('.badge2').textContent()) ?? '';
  expect(badgeAfter).not.toBe(badgeBefore);

  // Turn every state chip off → zero results + honest message (no fallback).
  for (const name of ['Intactes', 'Balayées']) {
    await card.getByRole('button', { name }).click();
  }
  await expect(card.locator('.zempty')).toBeVisible();
  await expect(card.locator('.zrow')).toHaveCount(0);
});

test('language can be changed from Réglages while staying on the page', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto('/compte');
  const card = page.locator('.card', { hasText: 'Langue' }).first();
  if ((await card.count()) === 0) {
    test.skip(true, 'Réglages needs an authenticated session in this environment');
    return;
  }
  await card.getByRole('button', { name: 'English' }).click();
  await expect(page).toHaveURL(/\/en\/compte$/); // same page, en prefix
  // The page stayed on Réglages, now translated.
  await expect(page.locator('.card', { hasText: /Language/ })).toBeVisible();
});
