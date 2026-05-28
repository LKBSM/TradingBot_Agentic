import { expect, test } from '@playwright/test';

test.describe('Landing — golden paths (architecture L1-L6 2026-05-27)', () => {
  test('brand visible in nav + hero badges (no marketing H1 above fold)', async ({ page }) => {
    await page.goto('/');
    // Brand in the nav.
    await expect(page.getByRole('link', { name: /MIA Markets/i })).toBeVisible();
    // SEO H1 is sr-only — present in DOM but visually hidden.
    await expect(
      page.getByRole('heading', { level: 1, name: /Comprenez le marché/i }),
    ).toBeAttached();
    // Hero eyebrow badges visible.
    await expect(page.getByText(/Lecture en direct/i)).toBeVisible();
    await expect(
      page.getByText(/Indicateur · pas un service de signaux/i),
    ).toBeVisible();
    // Discreet CTA (not a hero-dominant button anymore).
    await expect(
      page.getByRole('link', { name: /Essayer gratuitement/i }),
    ).toBeVisible();
  });

  test('multi-market section shows XAU + EUR + Bientôt placeholder', async ({ page }) => {
    await page.goto('/#multi-marche');
    await expect(
      page.getByRole('heading', {
        level: 2,
        name: /Sur chaque marché, le même cadre de lecture/i,
      }),
    ).toBeVisible();
    await expect(page.getByText(/BTC\/USD/i)).toBeVisible();
    await expect(page.getByText(/Bientôt/i).first()).toBeVisible();
  });

  test('conversations section auto-plays three replay tiles', async ({ page }) => {
    await page.goto('/#conversations');
    await expect(
      page.getByRole('heading', {
        level: 2,
        name: /Sentinel répond aux vraies questions/i,
      }),
    ).toBeVisible();
    // Three tile titles.
    await expect(page.getByText(/Décomposer un score/i)).toBeVisible();
    await expect(page.getByText(/Contextualiser un événement/i)).toBeVisible();
    await expect(page.getByText(/Refuser un ordre/i)).toBeVisible();
    // Compliance badge on the refusal tile.
    await expect(page.getByText(/Compliance/).first()).toBeVisible();
  });

  test('honest confidence section publishes the imposed citation + the bad numbers', async ({ page }) => {
    await page.goto('/#honnetete');
    // Imposed citation (lock 2 utilisateur 2026-05-27).
    await expect(
      page.getByText(/Aucun indicateur de marché ne devrait promettre des gains/i),
    ).toBeVisible();
    // PF 0,786 published.
    await expect(page.getByText(/0,786/)).toBeVisible();
    // Sub-perf vs buy-and-hold.
    await expect(page.getByText(/−318/)).toBeVisible();
  });

  test('pricing section shows 3 tiers FREE/9€/19€ post pivot 2026-05-27', async ({ page }) => {
    await page.goto('/#tarifs');
    await expect(page.getByText(/Découverte/)).toBeVisible();
    await expect(page.getByText(/Approfondie/)).toBeVisible();
    await expect(page.getByText(/Intégrale/)).toBeVisible();
    await expect(page.getByText(/9 €/).first()).toBeVisible();
    await expect(page.getByText(/19 €/).first()).toBeVisible();
    // INSTITUTIONAL retiré grille publique → bloc Calendly aside.
    await expect(page.getByText(/Réserver une démo/i)).toBeVisible();
    await expect(page.getByText(/Recommandé/)).toBeVisible();
  });

  test('FAQ accordion exposes 6 questions and opens the first one', async ({ page }) => {
    await page.goto('/#faq');
    await expect(
      page.getByRole('heading', { level: 2, name: /Vous vous demandez/i }),
    ).toBeVisible();
    const firstTrigger = page
      .getByRole('button', { name: /MIA est-il un service de signaux/i })
      .first();
    await firstTrigger.click();
    await expect(
      page.getByText(/analyses éditoriales contextuelles/i),
    ).toBeVisible();
  });

  test('footer lists 9 Phase-1 countries explicitly', async ({ page }) => {
    await page.goto('/');
    const footer = page.getByRole('contentinfo');
    await expect(footer.getByText(/France/)).toBeVisible();
    await expect(footer.getByText(/Belgique/)).toBeVisible();
    await expect(footer.getByText(/Canada \(hors Québec\)/i)).toBeVisible();
    await expect(footer.getByText(/Royaume-Uni/)).toBeVisible();
    await expect(footer.getByText(/Australie/)).toBeVisible();
    await expect(footer.getByText(/Nouvelle-Zélande/)).toBeVisible();
    await expect(footer.getByText(/Irlande/)).toBeVisible();
    // Early Access badge.
    await expect(footer.getByText(/Early Access · 50 places/i)).toBeVisible();
  });
});
