import { expect, test } from '@playwright/test';

/**
 * Product shell (mission UI-1) — golden paths on the /app surface. The shell is
 * the terminal-style frame (rail · center · docked chat); these checks assert it
 * renders, its rail navigation works, and the desktop grid never overflows
 * horizontally at the 1280px breakpoint.
 */
test.describe('Product shell — /app', () => {
  test('renders the rail and navigates via the ESPACE links', async ({ page }) => {
    await page.goto('/app');

    // The rail is present (its aria-label comes from app.sidebar.navAria).
    const rail = page.getByRole('complementary', { name: /combinaisons disponibles/i });
    await expect(rail).toBeVisible();

    // ESPACE nav routes to a sibling product surface.
    await rail.getByRole('link', { name: 'Zones' }).click();
    await expect(page).toHaveURL(/\/zones$/);
    // The rail persists across product routes (it lives in the group layout).
    await expect(
      page.getByRole('complementary', { name: /combinaisons disponibles/i }),
    ).toBeVisible();
  });

  test('has no horizontal overflow at the 1280px breakpoint', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('/app');
    await expect(
      page.getByRole('complementary', { name: /combinaisons disponibles/i }),
    ).toBeVisible();

    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    // Allow a 1px sub-pixel rounding tolerance.
    expect(overflow).toBeLessThanOrEqual(1);
  });
});
