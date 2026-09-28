import { expect, test } from '@playwright/test';

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
  test('renders the rail and the chat pedagogical disclaimer', async ({ page }) => {
    // Le rail est masqué sous 768px et /app n'a pas de `MobileSpaceNav` : ce
    // cas éprouve le châssis DE BUREAU et doit donc fixer sa fenêtre.
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('/app');

    // Shell rail (data-independent — part of the product shell).
    await expect(
      page.getByRole('complementary', { name: /combinaisons disponibles/i }),
    ).toBeVisible();

    // L'avertissement d'honnêteté doit être là. UI-3 (commit 290b8cf) a retiré
    // la ligne « Analyse pédagogique » de la barre de chat comme DOUBLON de la
    // ligne de conformité — cohérent avec CLN-1 §5, « un seul avertissement par
    // page ». Le mot « pédagogique » n'est donc plus rendu sur /app ; la
    // formulation en vigueur, verrouillée par cln-1-disclaimers.spec.ts, est
    // « Lecture algorithmique éducative ». La garantie est la même, seul le mot
    // a changé.
    await expect(
      page.getByText(/Lecture algorithmique éducative/i).first(),
    ).toBeVisible();
  });
});
