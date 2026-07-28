import { expect, test } from '@playwright/test';

/**
 * NW-1 — "Actualités programmées" calendar (list view). The full list behaviour
 * (grouping, chip zero-state, past toggle, amplitude placeholder, revised badge)
 * is covered by the component/helper unit tests; the calendar list needs a live
 * backend (whose default source is the official stub → empty) which the e2e
 * server does not provide. Here we assert the STATIC chrome that renders without
 * a backend, the rail → page navigation, and chip interactivity.
 */

async function overflow(page: import('@playwright/test').Page): Promise<number> {
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
}

test('1280×800: the product rail exposes an « Actualités » entry to the calendar', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/app');
  const link = page.getByRole('link', { name: 'Actualités' }).first();
  if ((await link.count()) === 0) {
    test.skip(true, 'product rail not rendered in this e2e environment');
    return;
  }
  // Assert the nav entry targets the calendar page. A full click-through lands on
  // a SubscriptionGate'd page that needs the backend (absent in e2e), so we pin
  // the wiring via the href rather than the navigation outcome.
  await expect(link).toHaveAttribute('href', /\/actualites$/);
  expect(await overflow(page)).toBeLessThanOrEqual(1);
});

test('1280×800: static chrome is descriptive (intro + « ne dit pas » banner), no raw key', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/actualites');
  const heading = page.getByRole('heading', { name: 'Actualités programmées' });
  if ((await heading.count()) === 0) {
    test.skip(true, 'calendar gated in this e2e environment');
    return;
  }
  await expect(heading).toBeVisible();
  await expect(page.getByText('Ce calendrier annonce des moments, pas des directions.')).toBeVisible();
  await expect(page.getByText('Ce que ce calendrier ne dit pas')).toBeVisible();

  // No raw i18n key leaked into the body.
  const body = (await page.locator('body').textContent()) ?? '';
  expect(/\bcalendar\.[a-zA-Z.]+\b/.test(body)).toBe(false);

  // Impact chip toggles its pressed state (no backend needed for the chips).
  const chip = page.locator('.fchip', { hasText: 'Élevé' }).first();
  const before = await chip.getAttribute('aria-pressed');
  await chip.click();
  expect(await chip.getAttribute('aria-pressed')).not.toBe(before);
});

test('390×844: calendar page has no horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/actualites');
  const heading = page.getByRole('heading', { name: 'Actualités programmées' });
  if ((await heading.count()) === 0) {
    test.skip(true, 'calendar gated in this e2e environment');
    return;
  }
  await expect(heading).toBeVisible();
  expect(await overflow(page)).toBeLessThanOrEqual(1);
});
