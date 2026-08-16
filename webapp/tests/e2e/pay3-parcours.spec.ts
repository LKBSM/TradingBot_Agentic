import { test, expect, type Page, type Route } from '@playwright/test';
import { dismissCookieBanner } from './utils';

/**
 * PAY-3 — the complete access journey, from the browser. Focuses on the defects
 * PAY-3 fixes (the six-state routing itself is covered by pay2-access.spec.ts):
 *
 *   A — a Google failure is VISIBLE on screen (never a silent bounce),
 *   B — the sign-up page SELLS (price + currency, three steps) and offers the
 *       Google door at parity, with no "free/trial" anywhere,
 *   C — login routes a returning UNSUBSCRIBED account to /abonnement,
 *       registration routes to the "confirm your email" screen (with resend),
 *   E — abandoning Checkout returns to a clean subscribe state, not an error.
 *
 * Both viewports (1280×800 + 390×844) run via the two Playwright projects. Each
 * test is run in French (default) and English (/en) where the copy matters.
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

/** Google is configured server-side → the button renders on both auth pages. */
async function googleEnabled(page: Page, enabled = true) {
  await page.route('**/api/auth/google/config', (r: Route) => r.fulfill(json({ enabled })));
}

// =============================================================================
// A — a Google failure is never silent
// =============================================================================

test.describe('A — Google failure is visible', () => {
  for (const { locale, path, re } of [
    { locale: 'fr', path: '/connexion?error=google&reason=state', re: /session a expiré|cookies sont bloqués/i },
    { locale: 'en', path: '/en/connexion?error=google&reason=exchange', re: /couldn.t confirm your sign-in/i },
  ]) {
    test(`${locale}: reason is rendered on screen`, async ({ page }) => {
      await googleEnabled(page);
      await page.goto(path);
      await dismissCookieBanner(page);
      // The banner is a clear, actionable message — not a blank form. (Target the
      // text, not role=alert: Next.js adds an empty route-announcer with the same
      // role, so getByRole('alert') is ambiguous.)
      await expect(page.getByText(re)).toBeVisible({ timeout: 15_000 });
    });
  }

  test('the reason param is cleaned from the URL after showing', async ({ page }) => {
    await googleEnabled(page);
    await page.goto('/connexion?error=google&reason=state');
    await dismissCookieBanner(page);
    await expect(page.getByText(/session a expiré|cookies sont bloqués/i)).toBeVisible({ timeout: 15_000 });
    await expect(page).not.toHaveURL(/error=google/);
  });
});

// =============================================================================
// B — the sign-up page sells + Google at parity + no "free/trial"
// =============================================================================

test.describe('B — sign-up page sells', () => {
  for (const { locale, path, price } of [
    { locale: 'fr', path: '/inscription', price: /39\s*\$\s*US\s*\/\s*mois/i },
    { locale: 'en', path: '/en/inscription', price: /39\s*USD\s*\/\s*month/i },
  ]) {
    test(`${locale}: shows price with currency and the Google door`, async ({ page }) => {
      await googleEnabled(page);
      await page.goto(path);
      await dismissCookieBanner(page);
      // Price with an explicit currency is visible (never a naked number).
      await expect(page.getByText(price).first()).toBeVisible({ timeout: 15_000 });
      // The Google door is present at sign-up too (parity with login).
      await expect(page.getByRole('link', { name: /google/i })).toBeVisible();
      // None of the forbidden words anywhere on the page (mission §5).
      const body = (await page.locator('body').innerText()).toLowerCase();
      expect(body).not.toMatch(/gratuit|essai|free trial/i);
    });
  }
});

// =============================================================================
// B/parity — Google door present at LOGIN too
// =============================================================================

test('Google door present on the login page', async ({ page }) => {
  await googleEnabled(page);
  await page.goto('/connexion');
  await dismissCookieBanner(page);
  await expect(page.getByRole('link', { name: /google/i })).toBeVisible({ timeout: 15_000 });
});

// =============================================================================
// C — registration routes to the "confirm your email" screen (with resend)
// =============================================================================

test('registration routes to the email-confirmation screen with a resend button', async ({ page }) => {
  await googleEnabled(page);
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
  await page.route('**/api/access/me', (r: Route) => r.fulfill(json({
    authenticated: true, gate_enforced: true, beta_lockdown: false, must_login: false,
    is_owner: false, has_access: false, email_verified: true,
    email_verification_required: false, subscription_required: true,
  })));
  await page.route('**/api/billing/pricing', (r: Route) => r.fulfill(json(PRICING)));
  await page.route('**/api/billing/subscription', (r: Route) => r.fulfill(json({ detail: 'no sub' }, 401)));
  await googleEnabled(page);

  await page.goto('/connexion');
  await dismissCookieBanner(page);
  await page.getByLabel(/nom d.utilisateur ou e-mail/i).fill('buyer@example.com');
  await page.getByLabel(/mot de passe/i).fill('longpassword1');
  await page.getByRole('button', { name: /se connecter/i }).click();

  await expect(page).toHaveURL(/\/abonnement(\?|$)/, { timeout: 15_000 });
});

// =============================================================================
// E — abandoning Checkout returns to a clean subscribe state, not an error
// =============================================================================

test('checkout abandon returns to /abonnement with a clean subscribe state', async ({ page }) => {
  await page.route('**/api/auth/me', (r: Route) => r.fulfill(json(ACCOUNT)));
  await page.route('**/api/access/me', (r: Route) => r.fulfill(json({
    authenticated: true, gate_enforced: true, beta_lockdown: false, must_login: false,
    is_owner: false, has_access: false, email_verified: true,
    email_verification_required: false, subscription_required: true,
  })));
  await page.route('**/api/billing/pricing', (r: Route) => r.fulfill(json(PRICING)));
  await page.route('**/api/billing/subscription', (r: Route) => r.fulfill(json({ detail: 'no sub' }, 401)));

  await page.goto('/abonnement?status=cancel');
  await dismissCookieBanner(page);
  // A plan is still offered (clean funnel, not an error page).
  await expect(page.getByRole('button', { name: /abonner|subscribe/i }).first()).toBeVisible({ timeout: 15_000 });
});
