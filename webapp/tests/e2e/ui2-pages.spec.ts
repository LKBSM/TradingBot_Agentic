import { expect, test } from '@playwright/test';
import frMessages from '@/messages/fr.json';

/**
 * UI-2 desktop pages — per-page smoke (rendered without error + key elements
 * visible). These assert only surfaces that render WITHOUT the data backend: the
 * fully-static Connexion card + animated canvas, and the shell chrome/chat on
 * /app. The data-dependent reference elements (App legalbar/panels, Scanner note,
 * Zones state badges, Réglages tiles) are covered by their component unit tests
 * (they need a live reading/session the e2e server does not provide).
 */

test.describe('UI-2 — Connexion (static card + candle canvas)', () => {
  test('renders the centered card, early-access badge, canvas and legal links', async ({ page }) => {
    await page.goto('/connexion');

    // Single centered card, no left marketing panel.
    await expect(page.locator('.login-card')).toBeVisible();
    await expect(page.locator('.login-card .ea')).toContainText(/Accès anticipé/i);

    // Animated candle-drift canvas sits behind the card.
    await expect(page.locator('canvas.fx')).toHaveCount(1);

    // "Rester connecté" is intentionally omitted (no remember-me in the API).
    await expect(page.getByText(/Rester connecté/i)).toHaveCount(0);

    // Mandatory legal links UNDER the card.
    const links = page.locator('.under-links a');
    await expect(links).toHaveCount(2);
    await expect(links.first()).toHaveAttribute('href', /conditions/);
    await expect(links.last()).toHaveAttribute('href', /confidentialite/);
  });

  test('reduced-motion draws a single static frame (canvas still present, no loop crash)', async ({ browser }) => {
    const ctx = await browser.newContext({ reducedMotion: 'reduce' });
    const page = await ctx.newPage();
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto('/connexion');
    await expect(page.locator('canvas.fx')).toBeVisible();
    expect(errors, 'no runtime error under reduced motion').toEqual([]);
    await ctx.close();
  });
});

test.describe('UI-2 — App shell + docked chat', () => {
  test('renders the rail and the honesty disclaimer', async ({ page }) => {
    // Le rail est une surface DESKTOP : sous 1280 la coquille bascule sur le
    // plan mobile, qui n'en a pas. Sans cette ligne le test tournait aussi dans
    // le projet iPhone 12 et y cherchait un élément qui n'existe pas.
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('/app');

    // Shell rail (data-independent — part of the product shell).
    await expect(
      page.getByRole('complementary', { name: /combinaisons disponibles/i }),
    ).toBeVisible();

    // La mention d'honnêteté. Elle visait `/pédagogique/`, que UI-3 (290b8cf,
    // 2026-08-26) a retirée du chat comme doublon : elle ne vit plus qu'UNE fois,
    // dans la coquille. C'est cet exemplaire-là que le produit montre.
    await expect(
      page.getByText(frMessages.legal.disclaimer.chart, { exact: false }).first(),
    ).toBeVisible();
  });
});
