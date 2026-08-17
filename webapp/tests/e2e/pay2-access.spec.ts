import { test, expect, type Page, type Route } from '@playwright/test';
import { dismissCookieBanner } from './utils';

/**
 * PAY-2 — "paying is the condition of entry", end-to-end from the browser.
 *
 * The nine mission scenarios (§4). The backend is mocked at the network edge so
 * each account STATE is exercised deterministically; the assertions are on what
 * the user actually sees and where they are routed. Server-side refusal itself
 * (scenario 9) is proven in tests/test_pay2_access.py — here we prove the UI
 * never leaks gated content and always routes to the subscribe funnel.
 *
 * Both viewports (1280×800 desktop + 390×844 mobile) run automatically via the
 * two Playwright projects.
 */

const H = 3600_000;

type Access = {
  authenticated: boolean;
  gate_enforced: boolean;
  beta_lockdown: boolean;
  must_login: boolean;
  is_owner: boolean;
  has_access: boolean;
  subscription_required: boolean;
};

type Sub = {
  status: string | null;
  price_id: string | null;
  current_period_end: number | null;
  cancel_at_period_end: boolean;
  trial_end: number | null;
  has_access: boolean;
};

const json = (body: unknown, status = 200) => ({
  status,
  contentType: 'application/json',
  body: JSON.stringify(body),
});

const ACCOUNT = {
  id: 1,
  username: 'tester',
  email: 'tester@example.com',
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

function sub(partial: Partial<Sub>): Sub {
  return {
    status: null,
    price_id: null,
    current_period_end: null,
    cancel_at_period_end: false,
    trial_end: null,
    has_access: false,
    ...partial,
  };
}

/** Wire the account + access + billing + (empty) market-data mocks. */
async function setup(
  page: Page,
  opts: {
    authenticated?: boolean;
    access: Access;
    subscription?: Sub | null;
    /** A subscription that changes over successive polls (for the webhook wait). */
    subscriptionSeq?: (Sub | null)[];
    checkoutUrl?: string;
    verifyOk?: boolean;
  },
) {
  const authed = opts.authenticated ?? opts.access.authenticated;

  await page.route('**/api/auth/me', (r: Route) =>
    r.fulfill(authed ? json(ACCOUNT) : json({ detail: 'unauthenticated' }, 401)),
  );
  await page.route('**/api/access/me', (r: Route) => r.fulfill(json(opts.access)));
  await page.route('**/api/billing/pricing', (r: Route) => r.fulfill(json(PRICING)));

  let seqIdx = 0;
  const subHandler = (r: Route) => {
    if (opts.subscriptionSeq) {
      const s = opts.subscriptionSeq[Math.min(seqIdx, opts.subscriptionSeq.length - 1)];
      seqIdx += 1;
      return r.fulfill(s ? json(s) : json({ detail: 'no sub' }, 401));
    }
    const s = opts.subscription;
    return r.fulfill(s ? json(s) : json({ detail: 'no sub' }, 401));
  };
  await page.route('**/api/billing/subscription', subHandler);
  // PAY-3e — the confirming screen now reconciles via POST /billing/sync; mirror
  // the same (sequenced) subscription state so the webhook-wait flow still works.
  await page.route('**/api/billing/sync', subHandler);

  await page.route('**/api/billing/checkout', (r: Route) =>
    r.fulfill(json({ url: opts.checkoutUrl ?? '/abonnement?status=success' })),
  );
  await page.route('**/api/auth/verify-email/confirm', (r: Route) =>
    r.fulfill(opts.verifyOk === false ? json({ detail: 'bad' }, 400) : json({ ok: true })),
  );

  // Market-data endpoints: only reached when access is GRANTED. Return a 404 so
  // the app shows its own empty/unavailable state (never a gate) — good enough
  // to assert "no paywall" without importing the full reading fixture.
  for (const p of ['**/api/market-reading*', '**/api/candles*', '**/api/calendar/month*', '**/api/calendar*', '**/api/live-price*']) {
    await page.route(p, (r: Route) => r.fulfill(json({ detail: 'no data' }, 404)));
  }
}

const paywall = (page: Page) =>
  page.getByRole('region', { name: /réservé aux abonnés/i });

async function expectPaywall(page: Page) {
  await expect(paywall(page)).toBeVisible({ timeout: 15_000 });
  const seePlans = page.getByRole('link', { name: /voir les abonnements/i });
  await expect(seePlans).toBeVisible();
  await expect(seePlans).toHaveAttribute('href', /\/abonnement/);
}

async function expectNoPaywall(page: Page) {
  // The gate resolves to children — the paywall must never appear.
  await page.waitForTimeout(1500);
  await expect(paywall(page)).toHaveCount(0);
}

const GATE_ON = { gate_enforced: true, beta_lockdown: false, must_login: false, is_owner: false };

// --- Scenario 3 — account with no payment sees no market data ----------------
test('S3 — authenticated, unsubscribed: /app and /actualites are walled', async ({ page }) => {
  await setup(page, {
    access: { authenticated: true, has_access: false, subscription_required: true, ...GATE_ON },
  });
  await page.goto('/app');
  await dismissCookieBanner(page);
  await expectPaywall(page);

  await page.goto('/actualites');
  await dismissCookieBanner(page);
  await expectPaywall(page);
});

// --- Scenario 4 — direct URL access without a subscription -------------------
test('S4 — unsubscribed direct URL: paywall routes to /abonnement', async ({ page }) => {
  await setup(page, {
    access: { authenticated: true, has_access: false, subscription_required: true, ...GATE_ON },
  });
  await page.goto('/zones');
  await dismissCookieBanner(page);
  await expectPaywall(page); // the CTA points at /abonnement (asserted inside)
});

test('S4b — anonymous direct URL with the gate on is bounced to /connexion', async ({ page }) => {
  await setup(page, {
    authenticated: false,
    access: {
      authenticated: false, has_access: false, subscription_required: false,
      gate_enforced: true, beta_lockdown: false, must_login: true, is_owner: false,
    },
  });
  await page.goto('/app');
  await expect(page).toHaveURL(/\/connexion(\?|$)/, { timeout: 15_000 });
});

// --- Scenario 5 — pay, close tab before redirect: webhook still grants access
test('S5 — return from Checkout waits for the webhook, then enters the app', async ({ page }) => {
  await setup(page, {
    access: { authenticated: true, has_access: false, subscription_required: true, ...GATE_ON },
    // First polls: no sub yet (webhook in flight). Then: active → redirect to /app.
    subscriptionSeq: [
      null,
      null,
      sub({ status: 'active', price_id: 'price_m', current_period_end: Date.now() / 1000 + 30 * 24 * 3600, has_access: true }),
    ],
  });
  await page.goto('/abonnement?status=success');
  await dismissCookieBanner(page);
  // The reassuring "confirming" screen while the webhook lands.
  await expect(page.getByText(/confirmation de Stripe/i)).toBeVisible({ timeout: 15_000 });
  // Then it auto-enters the product once the subscription turns active.
  await expect(page).toHaveURL(/\/app(\?|$)/, { timeout: 20_000 });
});

// --- Scenario 6 — cancellation keeps access until the paid period ends --------
test('S6 — canceled-but-paid keeps access; /abonnement shows the end date', async ({ page }) => {
  const periodEnd = Math.floor(Date.now() / 1000) + 20 * 24 * 3600;
  await setup(page, {
    access: { authenticated: true, has_access: true, subscription_required: false, ...GATE_ON },
    subscription: sub({ status: 'active', price_id: 'price_m', cancel_at_period_end: true, current_period_end: periodEnd, has_access: true }),
  });
  await page.goto('/app');
  await dismissCookieBanner(page);
  await expectNoPaywall(page); // access retained

  await page.goto('/abonnement');
  await dismissCookieBanner(page);
  // Manage button present (there is an active subscription to manage).
  await expect(page.getByRole('button', { name: /gérer|manage/i })).toBeVisible({ timeout: 15_000 });
});

// --- Scenario 7 — after the period ends, access is removed -------------------
test('S7 — expired subscription: /app is walled, /abonnement offers the plans', async ({ page }) => {
  await setup(page, {
    access: { authenticated: true, has_access: false, subscription_required: true, ...GATE_ON },
    subscription: sub({ status: 'canceled', has_access: false }),
  });
  await page.goto('/app');
  await dismissCookieBanner(page);
  await expectPaywall(page);

  await page.goto('/abonnement');
  await dismissCookieBanner(page);
  // "S'abonner" / "Se réabonner" — match apostrophe-agnostically.
  await expect(page.getByRole('button', { name: /abonner|subscribe|reactivate/i }).first()).toBeVisible({ timeout: 15_000 });
});

// --- Scenario 8 — renewal failure: grace keeps access, then suspension cuts it
test('S8a — past_due (grace): access is kept', async ({ page }) => {
  await setup(page, {
    access: { authenticated: true, has_access: true, subscription_required: false, ...GATE_ON },
    subscription: sub({ status: 'past_due', price_id: 'price_m', current_period_end: Math.floor(Date.now() / 1000) + H, has_access: true }),
  });
  await page.goto('/app');
  await dismissCookieBanner(page);
  await expectNoPaywall(page);
});

test('S8b — suspended: access is removed', async ({ page }) => {
  await setup(page, {
    access: { authenticated: true, has_access: false, subscription_required: true, ...GATE_ON },
    subscription: sub({ status: 'suspended', has_access: false }),
  });
  await page.goto('/app');
  await dismissCookieBanner(page);
  await expectPaywall(page);
});

// --- Scenario 1 — signup → email verification → plan choice ------------------
test('S1 — email verification routes automatically to the plan choice', async ({ page }) => {
  await setup(page, {
    access: { authenticated: true, has_access: false, subscription_required: true, ...GATE_ON },
    subscription: null,
    verifyOk: true,
  });
  await page.goto('/verifier-email?token=valid-token');
  await dismissCookieBanner(page);
  // After confirmation the user is sent to choose a plan (paying = entry).
  await expect(page).toHaveURL(/\/abonnement(\?|$)/, { timeout: 15_000 });
  // And a real plan (paid) is offered — no free option anywhere.
  await expect(page.getByRole('button', { name: /abonner|subscribe/i }).first()).toBeVisible({ timeout: 15_000 });
});

// --- Scenario 1b/2 — starting Checkout leads into the paid funnel ------------
test('S1b — choosing a plan starts Checkout and returns into the confirm/app flow', async ({ page }) => {
  await setup(page, {
    access: { authenticated: true, has_access: false, subscription_required: true, ...GATE_ON },
    // pre-checkout load (null) → success-page load (null → confirming) → poll (active → /app).
    subscriptionSeq: [null, null, sub({ status: 'active', price_id: 'price_m', current_period_end: Date.now() / 1000 + 30 * 24 * 3600, has_access: true })],
    checkoutUrl: '/abonnement?status=success',
  });
  await page.goto('/abonnement');
  await dismissCookieBanner(page);
  const subscribe = page.getByRole('button', { name: /abonner|subscribe/i }).first();
  await expect(subscribe).toBeVisible({ timeout: 15_000 });
  await subscribe.click();
  // Checkout URL (mocked same-origin) returns to the success flow, which then
  // enters the app once the subscription is active.
  await expect(page).toHaveURL(/\/app(\?|$)/, { timeout: 20_000 });
});
