import { test, expect, type Page, type Route } from '@playwright/test';
import { dismissCookieBanner } from './utils';

/**
 * PAY-3 — LES SIX ÉTATS D'ACCÈS, un écran chacun.
 *
 * La règle de la mission : « Aucun état ne doit afficher un écran d'un autre
 * état. » Un test qui vérifie seulement que le bon écran est là ne prouve pas
 * ça — il faut AUSSI prouver que les autres écrans sont absents. Chaque cas
 * ci-dessous fait les deux : l'écran attendu est visible, et les marqueurs des
 * autres états ne le sont pas.
 *
 *   1. visiteur ............... /abonnement renvoie à /connexion
 *   2. connecté jamais payé ... choix de formule, pas de gestion
 *   3. paiement en confirmation attente du webhook, pas de formule à choisir
 *   4. abonné actif ........... gestion + prochain prélèvement
 *   5. résilié, non écoulé .... accès jusqu'à la date, sans renouvellement
 *   6. expiré ................. réactivation
 *   7. hors zone (G2) ......... refus expliqué, AUCUN appel à repayer
 *
 * Le 7e n'est pas un sixième état déguisé : un paiement hors Canada/États-Unis
 * est annulé et remboursé, donc renvoyer ce client vers les formules
 * l'inviterait à un paiement qu'on refusera encore.
 *
 * Les deux gabarits (1280×800 et 390×844) viennent des deux projets Playwright.
 */

const json = (body: unknown, status = 200) => ({
  status,
  contentType: 'application/json',
  body: JSON.stringify(body),
});

const ACCOUNT = {
  id: 1,
  username: 'buyer',
  email: 'buyer@example.com',
  role: 'user' as const,
  age_confirmed: true,
  email_verified: true,
  created_at: '2026-01-01T00:00:00Z',
  consents: [],
};

const PRICING = {
  plans: [
    { key: 'MONTHLY', price_id: 'price_m' },
    { key: 'ANNUAL', price_id: 'price_a' },
  ],
  trial_days: 0,
};

const accessMe = (hasAccess: boolean) => ({
  authenticated: true,
  gate_enforced: true,
  beta_lockdown: false,
  must_login: false,
  is_owner: false,
  has_access: hasAccess,
  email_verified: true,
  email_verification_required: false,
  subscription_required: !hasAccess,
});

const IN_30_DAYS = Math.floor(Date.now() / 1000) + 30 * 86400;

/** Wire the account + pricing mocks every state needs. */
async function mockAccount(page: Page, { hasAccess }: { hasAccess: boolean }) {
  await page.route('**/api/auth/me', (r: Route) => r.fulfill(json(ACCOUNT)));
  await page.route('**/api/access/me', (r: Route) => r.fulfill(json(accessMe(hasAccess))));
  await page.route('**/api/billing/pricing', (r: Route) => r.fulfill(json(PRICING)));
}

/** The subscription payload that decides which screen renders. */
async function mockSubscription(page: Page, body: unknown, status = 200) {
  await page.route('**/api/billing/subscription', (r: Route) => r.fulfill(json(body, status)));
  // The confirming screen reconciles through /sync (PAY-3e); keep it consistent
  // so a state can never drift mid-test.
  await page.route('**/api/billing/sync', (r: Route) => r.fulfill(json(body, status === 401 ? 200 : status)));
}

// Markers of each screen. Absence assertions are built from these, so adding a
// state here forces every other test to prove it does NOT show it.
const planChoice = (page: Page) => page.getByRole('button', { name: /abonner/i }).first();
const manageButton = (page: Page) => page.getByRole('button', { name: /Gérer mon abonnement/i });
const confirming = (page: Page) => page.locator('[aria-busy="true"]');
const blockedScreen = (page: Page) => page.getByTestId('subscription-blocked-region');

async function openSubscriptionPage(page: Page, query = '') {
  await page.goto(`/abonnement${query}`);
  await dismissCookieBanner(page);
}

// =============================================================================
// 1 — visiteur : pas d'écran d'abonnement du tout
// =============================================================================

test('1. visiteur — /abonnement renvoie à la connexion', async ({ page }) => {
  await page.route('**/api/auth/me', (r: Route) => r.fulfill(json({ detail: 'anonymous' }, 401)));
  await page.route('**/api/access/me', (r: Route) =>
    r.fulfill(
      json({
        authenticated: false,
        gate_enforced: true,
        beta_lockdown: false,
        must_login: true,
        is_owner: false,
        has_access: false,
        email_verified: false,
        email_verification_required: false,
        subscription_required: true,
      }),
    ),
  );
  await page.route('**/api/billing/pricing', (r: Route) => r.fulfill(json(PRICING)));

  await openSubscriptionPage(page);
  await expect(page).toHaveURL(/\/connexion/, { timeout: 15_000 });
  await expect(planChoice(page)).toBeHidden();
  await expect(manageButton(page)).toBeHidden();
});

// =============================================================================
// 2 — connecté, jamais payé : le choix de formule, et rien d'autre
// =============================================================================

test('2. connecté jamais payé — choix de formule, aucune gestion', async ({ page }) => {
  await mockAccount(page, { hasAccess: false });
  await mockSubscription(page, { detail: 'no sub' }, 401);

  await openSubscriptionPage(page);
  await expect(page.getByText('Active ton compte')).toBeVisible({ timeout: 15_000 });
  await expect(planChoice(page)).toBeVisible();

  // …et surtout : aucun écran d'un autre état.
  await expect(manageButton(page)).toBeHidden();
  await expect(confirming(page)).toBeHidden();
  await expect(blockedScreen(page)).toBeHidden();
  await expect(page.getByText('Réactive ton accès')).toBeHidden();
});

// =============================================================================
// 3 — paiement en cours de confirmation : l'attente, pas les formules
// =============================================================================

test('3. paiement en confirmation — attente du webhook, pas de formules', async ({ page }) => {
  await mockAccount(page, { hasAccess: false });
  await mockSubscription(page, { detail: 'no sub' }, 401);

  await openSubscriptionPage(page, '?status=success');
  await expect(confirming(page)).toBeVisible({ timeout: 15_000 });

  // Proposer une formule ici, c'est risquer un second paiement pour le même mois.
  await expect(planChoice(page)).toBeHidden();
  await expect(manageButton(page)).toBeHidden();
  await expect(blockedScreen(page)).toBeHidden();
});

// =============================================================================
// 4 — abonné actif : la gestion, jamais l'invitation à s'abonner
// =============================================================================

test('4. abonné actif — gestion et prochain prélèvement', async ({ page }) => {
  await mockAccount(page, { hasAccess: true });
  await mockSubscription(page, {
    status: 'active',
    price_id: 'price_m',
    current_period_end: IN_30_DAYS,
    cancel_at_period_end: false,
    trial_end: null,
    has_access: true,
  });

  await openSubscriptionPage(page);
  await expect(page.getByText('Abonnement actif')).toBeVisible({ timeout: 15_000 });
  await expect(manageButton(page)).toBeVisible();

  await expect(planChoice(page)).toBeHidden();
  await expect(confirming(page)).toBeHidden();
  await expect(blockedScreen(page)).toBeHidden();
  await expect(page.getByText('Active ton compte')).toBeHidden();
});

// =============================================================================
// 5 — résilié, période non écoulée : l'accès court encore, sans renouvellement
// =============================================================================

test('5. résilié période non écoulée — accès jusqu\'à la date, sans renouvellement', async ({ page }) => {
  await mockAccount(page, { hasAccess: true });
  await mockSubscription(page, {
    status: 'active',
    price_id: 'price_m',
    current_period_end: IN_30_DAYS,
    cancel_at_period_end: true,
    trial_end: null,
    has_access: true,
  });

  await openSubscriptionPage(page);
  await expect(page.getByText(/sans renouvellement/i)).toBeVisible({ timeout: 15_000 });
  await expect(manageButton(page)).toBeVisible();

  // Un abonné résilié ne doit PAS lire « prochain prélèvement ».
  await expect(page.getByText(/Prochain prélèvement/i)).toBeHidden();
  await expect(planChoice(page)).toBeHidden();
  await expect(blockedScreen(page)).toBeHidden();
});

// =============================================================================
// 6 — expiré : la réactivation, pas la gestion
// =============================================================================

test('6. expiré — écran de réactivation', async ({ page }) => {
  await mockAccount(page, { hasAccess: false });
  await mockSubscription(page, {
    status: 'canceled',
    price_id: 'price_m',
    current_period_end: Math.floor(Date.now() / 1000) - 86400,
    cancel_at_period_end: false,
    trial_end: null,
    has_access: false,
  });

  await openSubscriptionPage(page);
  await expect(page.getByText('Réactive ton accès')).toBeVisible({ timeout: 15_000 });
  await expect(planChoice(page)).toBeVisible();

  await expect(manageButton(page)).toBeHidden();
  await expect(confirming(page)).toBeHidden();
  await expect(blockedScreen(page)).toBeHidden();
  await expect(page.getByText('Active ton compte')).toBeHidden();
});

// =============================================================================
// 7 — hors zone (G2) : le refus expliqué, AUCUN appel à repayer
// =============================================================================

test('7. hors zone — refus expliqué, aucune formule proposée', async ({ page }) => {
  await mockAccount(page, { hasAccess: false });
  await mockSubscription(page, {
    status: 'blocked_region',
    price_id: null,
    current_period_end: null,
    cancel_at_period_end: false,
    trial_end: null,
    has_access: false,
  });

  await openSubscriptionPage(page);
  await expect(blockedScreen(page)).toBeVisible({ timeout: 15_000 });
  // La zone est NOMMÉE à l'écran : un refus sans motif est un mur, pas une réponse.
  await expect(page.getByText(/vendu au Canada/).first()).toBeVisible();

  // Le cœur du test : on ne réinvite pas à payer ce qu'on refusera encore.
  await expect(planChoice(page)).toBeHidden();
  await expect(manageButton(page)).toBeHidden();
  await expect(confirming(page)).toBeHidden();
  await expect(page.getByText('Réactive ton accès')).toBeHidden();
  await expect(page.getByText('Active ton compte')).toBeHidden();
});

// =============================================================================
// Le retour de Checkout hors zone ne doit pas tourner en boucle
// =============================================================================

test('hors zone au retour de Checkout — pas de spinner sans fin', async ({ page }) => {
  await mockAccount(page, { hasAccess: false });
  await mockSubscription(page, {
    status: 'blocked_region',
    price_id: null,
    current_period_end: null,
    cancel_at_period_end: false,
    trial_end: null,
    has_access: false,
  });

  await openSubscriptionPage(page, '?status=success');
  await expect(blockedScreen(page)).toBeVisible({ timeout: 15_000 });
  await expect(confirming(page)).toBeHidden();
});
