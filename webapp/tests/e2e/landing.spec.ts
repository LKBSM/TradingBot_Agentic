import { expect, test } from '@playwright/test';
import { dismissCookieBanner } from './utils';

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
      page.getByRole('link', { name: /Voir l'offre/i }),
    ).toBeVisible();
  });

  test('how-it-works section defines the product in 4 steps', async ({ page }) => {
    await page.goto('/#fonctionnement');
    await expect(
      page.getByRole('heading', {
        level: 2,
        name: /De la bougie brute à une lecture/i,
      }),
    ).toBeVisible();
    await expect(page.getByText(/Ce que les démos prouvent/i)).toBeVisible();
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
        name: /M\.I\.A Agent répond aux vraies questions/i,
      }),
    ).toBeVisible();
    // Three tile titles (5.D — "score" reformulated to a descriptive reading).
    await expect(page.getByText(/Comprendre une lecture/i)).toBeVisible();
    await expect(page.getByText(/Contextualiser un événement/i)).toBeVisible();
    await expect(page.getByText(/Refuser un ordre/i)).toBeVisible();
    // Compliance badge on the refusal tile.
    await expect(page.getByText(/Compliance/).first()).toBeVisible();
  });

  test('honest confidence section keeps the imposed citation + engagement, drops pre-pivot backtest numbers', async ({ page }) => {
    await page.goto('/#honnetete');
    // Imposed citation (lock 2 utilisateur 2026-05-27) — conservée en 5.C.
    await expect(
      page.getByText(/Aucun indicateur de marché ne devrait promettre des gains/i),
    ).toBeVisible();
    // « Ce que nous ne ferons jamais » column conservée.
    await expect(page.getByText(/Ce que nous ne ferons jamais/i)).toBeVisible();
    // Chantier 5.C : résidus positionnement pré-pivot retirés (backtest PF,
    // sous-perf, DSR/PBO, colonne « edge mesurable »).
    await expect(page.getByText(/0,786/)).toHaveCount(0);
    await expect(page.getByText(/−318/)).toHaveCount(0);
    await expect(page.getByText(/Deflated Sharpe/i)).toHaveCount(0);
    await expect(page.getByText(/edge mesurable/i)).toHaveCount(0);
    // Chantier 5.D : vulgarisation — plus de jargon "conformel(le)" visible.
    await expect(page.getByText(/conformel/i)).toHaveCount(0);
  });

  test('pricing section shows the single plan with monthly/annual toggle', async ({ page }) => {
    await page.goto('/#tarifs');
    await expect(page.getByRole('heading', { name: /Un seul plan/i })).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'Accès intégral MIA' }),
    ).toBeVisible();
    // Annual is the default cadence → 39,99 $ shown.
    await expect(page.getByText(/39,99/).first()).toBeVisible();
    // Switching to monthly reveals 49,99 $.
    await page.getByRole('button', { name: /^Mensuel$/i }).click();
    await expect(page.getByText(/49,99/).first()).toBeVisible();
    // Old tiers are gone.
    await expect(page.getByText(/Approfondie/)).toHaveCount(0);
    await expect(page.getByText(/Intégrale/)).toHaveCount(0);
    // B2B contact block conservé.
    await expect(page.getByText(/Réserver une démo/i)).toBeVisible();
  });

  test('FAQ accordion exposes 6 questions and opens the first one', async ({ page }) => {
    await page.goto('/#faq');
    // Mobile : la bannière cookies intercepte les clics près du bord bas.
    await dismissCookieBanner(page);
    await expect(
      page.getByRole('heading', { level: 2, name: /Vous vous demandez/i }),
    ).toBeVisible();
    const firstTrigger = page
      .getByRole('button', { name: /MIA est-il un service de signaux/i })
      .first();
    await firstTrigger.click();
    // Texte de la réponse q1 (post nettoyage claims 2026-07-04).
    await expect(
      page.getByText(/refuse explicitement les questions/i),
    ).toBeVisible();
  });

  test('footer only states verifiable facts (claims cleanup 2026-07-04)', async ({ page }) => {
    await page.goto('/');
    const footer = page.getByRole('contentinfo');
    // Honest educational disclaimer stays.
    await expect(footer.getByText(/Lecture algorithmique éducative/i)).toBeVisible();
    await expect(footer.getByText(/ni un signal de trading/i)).toBeVisible();
    // Live legal links only (targets exist in the repo).
    await expect(footer.getByRole('link', { name: /Conditions d.utilisation/i })).toBeVisible();
    await expect(footer.getByRole('link', { name: /Confidentialité/i })).toBeVisible();
    // Removed false / unverified claims must NOT reappear anywhere on the page.
    const body = page.locator('body');
    await expect(body.getByText(/2024\/2811/)).toHaveCount(0);
    await expect(body.getByText(/CM2C/)).toHaveCount(0);
    await expect(body.getByText(/50 places/)).toHaveCount(0);
    await expect(body.getByText(/hors Québec/i)).toHaveCount(0);
    await expect(body.getByText(/Médiateur/i)).toHaveCount(0);
    // Dead links removed from the footer.
    await expect(footer.getByRole('link', { name: /Mentions légales/i })).toHaveCount(0);
    await expect(footer.getByRole('link', { name: /Cookies/i })).toHaveCount(0);
  });
});
