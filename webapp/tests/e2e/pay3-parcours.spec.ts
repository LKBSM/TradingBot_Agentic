import { test, expect, type Page, type Route } from '@playwright/test';
import { dismissCookieBanner } from './utils';

/**
 * PAY-3 / PAY-3b — the access journey, email-only (Google removed), from the
 * browser:
 *
 *   B — the sign-up page SELLS (price + currency, three steps), email-only,
 *       with NO Google door and no "free/trial" copy,
 *   C — registration routes to the "confirm your email" screen (with resend),
 *       login routes a returning UNSUBSCRIBED account to /abonnement,
 *   D — the /abonnement view for an unsubscribed account is a clean plan choice
 *       (no "no subscription" status), paying being the only door in,
 *   E — abandoning Checkout returns to a clean subscribe state, not an error.
 *
 * Both viewports (1280×800 + 390×844) run via the two Playwright projects.
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

const PRICING = { plans: [{ key: 'MONTHLY', price_id: 'price_m' }, { key: 'ANNUAL', price_id: 'price_a' }], trial_days: 0 };

const NO_ACCESS = {
  authenticated: true, gate_enforced: true, beta_lockdown: false, must_login: false,
  is_owner: false, has_access: false, email_verified: true,
  email_verification_required: false, subscription_required: true,
};

// =============================================================================
// B — the sign-up page sells, email-only, no Google, no "free/trial"
// =============================================================================

test.describe('B — sign-up page sells (email-only)', () => {
  for (const { locale, path, price } of [
    { locale: 'fr', path: '/inscription', price: /39\s*\$\s*US\s*\/\s*mois/i },
    { locale: 'en', path: '/en/inscription', price: /39\s*USD\s*\/\s*month/i },
  ]) {
    test(`${locale}: shows price with currency, no Google, no free/trial`, async ({ page }) => {
      await page.goto(path);
      await dismissCookieBanner(page);
      // Price with an explicit currency is visible (never a naked number).
      await expect(page.getByText(price).first()).toBeVisible({ timeout: 15_000 });
      // Google door is GONE — email/password only.
      await expect(page.getByRole('link', { name: /google/i })).toHaveCount(0);
      await expect(page.getByRole('button', { name: /google/i })).toHaveCount(0);
      // No forbidden words anywhere on the page (mission §5).
      const body = (await page.locator('body').innerText()).toLowerCase();
      expect(body).not.toMatch(/gratuit|essai|free trial/i);
    });
  }
});

test('no Google door on the login page either', async ({ page }) => {
  await page.goto('/connexion');
  await dismissCookieBanner(page);
  await expect(page.getByRole('link', { name: /se connecter/i }).or(page.getByRole('button', { name: /se connecter/i })).first()).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole('link', { name: /google/i })).toHaveCount(0);
  await expect(page.getByRole('button', { name: /google/i })).toHaveCount(0);
});

// =============================================================================
// C — registration routes to the "confirm your email" screen (with resend)
// =============================================================================

test('registration routes to the email-confirmation screen with a resend button', async ({ page }) => {
  await page.route('**/api/auth/register', (r: Route) => r.fulfill(json(ACCOUNT, 201)));
  await page.route('**/api/auth/me', (r: Route) => r.fulfill(json(ACCOUNT)));
  await page.route('**/api/auth/verify-email/resend', (r: Route) => r.fulfill(json({ ok: true })));

  await page.goto('/inscription');
  await dismissCookieBanner(page);
  await page.getByLabel(/adresse e-mail/i).fill('buyer@example.com');
  await page.getByLabel(/mot de passe/i).fill('longpassword1');
  await page.getByLabel(/18 ans/i).check();
  await page.getByLabel(/conditions/i).check();
  await page.getByLabel(/confidentialité/i).check();
  await page.getByRole('button', { name: /créer mon compte/i }).click();

  await expect(page).toHaveURL(/\/verifier-email(\?|$)/, { timeout: 15_000 });
  await expect(page.getByText(/confirme ton adresse|boîte de réception/i)).toBeVisible();
  await expect(page.getByRole('button', { name: /renvoyer l.e-mail/i })).toBeVisible();
});

// =============================================================================
// C — a returning UNSUBSCRIBED account logging in is routed to /abonnement
// =============================================================================

test('login of an unsubscribed account routes to /abonnement, not /app', async ({ page }) => {
  await page.route('**/api/auth/login', (r: Route) => r.fulfill(json(ACCOUNT)));
  await page.route('**/api/auth/me', (r: Route) => r.fulfill(json(ACCOUNT)));
  await page.route('**/api/access/me', (r: Route) => r.fulfill(json(NO_ACCESS)));
  await page.route('**/api/billing/pricing', (r: Route) => r.fulfill(json(PRICING)));
  await page.route('**/api/billing/subscription', (r: Route) => r.fulfill(json({ detail: 'no sub' }, 401)));

  await page.goto('/connexion');
  await dismissCookieBanner(page);
  await page.getByLabel(/nom d.utilisateur ou e-mail/i).fill('buyer@example.com');
  await page.getByLabel(/mot de passe/i).fill('longpassword1');
  await page.getByRole('button', { name: /se connecter/i }).click();

  await expect(page).toHaveURL(/\/abonnement(\?|$)/, { timeout: 15_000 });
});

// =============================================================================
// D — the unsubscribed /abonnement view is a clean plan choice (no "no sub")
// =============================================================================

test('unsubscribed /abonnement shows plan cards and no "no subscription" status', async ({ page }) => {
  await page.route('**/api/auth/me', (r: Route) => r.fulfill(json(ACCOUNT)));
  await page.route('**/api/access/me', (r: Route) => r.fulfill(json(NO_ACCESS)));
  await page.route('**/api/billing/pricing', (r: Route) => r.fulfill(json(PRICING)));
  await page.route('**/api/billing/subscription', (r: Route) => r.fulfill(json({ detail: 'no sub' }, 401)));

  await page.goto('/abonnement');
  await dismissCookieBanner(page);
  // The "activate your account" hero + two plan cards with a subscribe CTA.
  await expect(page.getByRole('heading', { name: /active ton compte/i })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole('button', { name: /abonner/i })).toHaveCount(2);
  await expect(page.getByText(/meilleure offre/i)).toBeVisible();
  // The "no subscription" resting state must be GONE.
  await expect(page.getByText(/aucun abonnement/i)).toHaveCount(0);
});

// =============================================================================
// E — abandoning Checkout returns to a clean subscribe state, not an error
// =============================================================================

test('checkout abandon returns to /abonnement with a clean subscribe state', async ({ page }) => {
  await page.route('**/api/auth/me', (r: Route) => r.fulfill(json(ACCOUNT)));
  await page.route('**/api/access/me', (r: Route) => r.fulfill(json(NO_ACCESS)));
  await page.route('**/api/billing/pricing', (r: Route) => r.fulfill(json(PRICING)));
  await page.route('**/api/billing/subscription', (r: Route) => r.fulfill(json({ detail: 'no sub' }, 401)));

  await page.goto('/abonnement?status=cancel');
  await dismissCookieBanner(page);
  // A plan is still offered (clean funnel, not an error page).
  await expect(page.getByRole('button', { name: /abonner/i }).first()).toBeVisible({ timeout: 15_000 });
});
