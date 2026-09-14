import { expect, test, type Page } from '@playwright/test';
import { dismissCookieBanner } from './utils';

/**
 * LEG-1 — the two legal pages and the consent screen, at both viewports.
 *
 * What is asserted here, and why it needs a browser rather than a unit test:
 *
 *  * /conditions and /confidentialite render the REAL document (fetched from
 *    the backend and rendered as markdown) and open WITHOUT an account — the
 *    consent screen links to them, so a visitor must be able to read them;
 *  * each page shows the version it is serving;
 *  * exactly ONE legal/educational disclaimer block per page (existing rule);
 *  * on /abonnement the box is not pre-ticked, both plan CTAs are inactive
 *    while it is unticked, the reason is written out, and ticking releases them.
 */

const VIEWPORTS = [
  { name: 'desktop 1280×800', width: 1280, height: 800 },
  { name: 'mobile 390×844', width: 390, height: 844 },
] as const;

const VERSION = '2026-09-13';

const TERMS_MD = `# M.I.A Markets — Conditions d'utilisation

_Version : ${VERSION} · Dernière mise à jour : 13 septembre 2026_

## 1. Ce qu'est ce service

Le service décrit ce qu'il observe. Il ne recommande rien.

## 8. Résiliation et remboursement

Tu peux résilier à tout moment, aussi simplement que tu t'es abonné.
`;

const PRIVACY_MD = `# M.I.A Markets — Politique de confidentialité

_Version : ${VERSION} · Dernière mise à jour : 13 septembre 2026_

## 1. Qui est responsable de tes renseignements

Loukmane Bessam — contact@mia.markets.

## 4. Où ils sont hébergés

Nos serveurs sont situés aux États-Unis.
`;

/** The legal documents come from the backend; serve them without one running. */
async function mockLegal(page: Page) {
  await page.route('**/api/v1/legal/conditions**', (r) =>
    r.fulfill({
      status: 200,
      headers: { 'content-type': 'text/markdown; charset=utf-8', 'x-document-version': VERSION },
      body: TERMS_MD,
    }),
  );
  await page.route('**/api/v1/legal/privacy**', (r) =>
    r.fulfill({
      status: 200,
      headers: { 'content-type': 'text/markdown; charset=utf-8', 'x-document-version': VERSION },
      body: PRIVACY_MD,
    }),
  );
}

/** A logged-in account with no subscription — the state that sees the gate. */
async function mockSubscribed(page: Page) {
  await page.route('**/api/access/me', (r) =>
    r.fulfill({
      json: {
        authenticated: true, gate_enforced: true, beta_lockdown: false,
        must_login: false, is_owner: false, has_access: false,
        subscription_required: true,
      },
    }),
  );
  await page.route('**/api/auth/me', (r) =>
    r.fulfill({
      json: {
        id: 2, username: 'buyer', email: 'buyer@example.com', role: 'user',
        age_confirmed: true, email_verified: true,
        created_at: '2026-09-01T10:00:00', consents: [],
      },
    }),
  );
  await page.route('**/api/billing/pricing', (r) =>
    r.fulfill({
      json: {
        plans: [
          { key: 'MONTHLY', price_id: 'price_m', amount_usd: 39 },
          { key: 'ANNUAL', price_id: 'price_a', amount_usd: 348 },
        ],
      },
    }),
  );
  await page.route('**/api/billing/subscription', (r) => r.fulfill({ json: null }));
  await page.route('**/api/billing/checkout', (r) =>
    r.fulfill({ json: { url: 'https://checkout.stripe.test/s/leg1' } }),
  );
}

/** Count VISIBLE elements carrying the page-disclaimer stem (CLN-1 §5 rule). */
async function visibleDisclaimers(page: Page): Promise<number> {
  const loc = page.locator('p, div', { hasText: 'Lecture algorithmique éducative' });
  let n = 0;
  for (const el of await loc.all()) {
    // Count only the leaf carrying the text, not its ancestors.
    const owns = await el.evaluate(
      (node) =>
        Array.from(node.childNodes).some(
          (c) => c.nodeType === Node.TEXT_NODE && (c.textContent ?? '').includes('Lecture algorithmique éducative'),
        ),
    );
    if (owns && (await el.isVisible())) n += 1;
  }
  return n;
}

for (const vp of VIEWPORTS) {
  test.describe(`LEG-1 @ ${vp.name}`, () => {
    test.use({ viewport: { width: vp.width, height: vp.height } });

    test('/conditions renders the real document, without an account', async ({ page }) => {
      await mockLegal(page);
      await page.goto('/conditions');
      await dismissCookieBanner(page);

      await expect(
        page.getByRole('heading', { level: 1, name: /Conditions d'utilisation/i }),
      ).toBeVisible();
      // The version served is displayed — not hard-coded in the page.
      await expect(page.getByText(`Version ${VERSION}`)).toBeVisible();
      // A real clause, rendered from the markdown.
      await expect(
        page.getByRole('heading', { level: 2, name: /Résiliation et remboursement/i }),
      ).toBeVisible();
      // Never the loading or error state. Scoped to the document container:
      // `next dev` injects its own dev-tools alert into the page, which is not
      // ours and must not decide this assertion.
      await expect(page.locator('.container-prose [role="alert"]')).toHaveCount(0);
    });

    test('/confidentialite renders the real document, not a placeholder', async ({ page }) => {
      await mockLegal(page);
      await page.goto('/confidentialite');
      await dismissCookieBanner(page);

      await expect(
        page.getByRole('heading', { level: 1, name: /Politique de confidentialité/i }),
      ).toBeVisible();
      await expect(page.getByText(`Version ${VERSION}`)).toBeVisible();
      await expect(
        page.getByRole('heading', { level: 2, name: /Où ils sont hébergés/i }),
      ).toBeVisible();
      // The "preliminary version" placeholder is gone for good.
      await expect(page.locator('[data-legal-pending="privacy-placeholder"]')).toHaveCount(0);
    });

    test('each legal page carries exactly ONE disclaimer block', async ({ page }) => {
      await mockLegal(page);
      for (const route of ['/conditions', '/confidentialite']) {
        await page.goto(route);
        await dismissCookieBanner(page);
        expect(await visibleDisclaimers(page), `${route} @ ${vp.name}`).toBe(1);
      }
    });

    test('/abonnement gates both payment CTAs behind an unticked box', async ({ page }) => {
      await mockLegal(page);
      await mockSubscribed(page);
      await page.goto('/abonnement');
      await dismissCookieBanner(page);

      const box = page.getByTestId('consent-checkbox');
      await expect(box).toBeVisible();
      await expect(box).not.toBeChecked();

      const ctas = page.locator('.grid button');
      await expect(ctas).toHaveCount(2);
      for (let i = 0; i < 2; i += 1) await expect(ctas.nth(i)).toBeDisabled();

      // The reason is WRITTEN, not left to a greyed button.
      await expect(page.getByTestId('consent-blocked')).toBeVisible();

      // Both documents are reachable from this very screen.
      await expect(page.locator('a[href*="/conditions"]').first()).toBeVisible();
      await expect(page.locator('a[href*="/confidentialite"]').first()).toBeVisible();

      await box.check();
      for (let i = 0; i < 2; i += 1) await expect(ctas.nth(i)).toBeEnabled();
      await expect(page.getByTestId('consent-blocked')).toHaveCount(0);
    });
  });
}
