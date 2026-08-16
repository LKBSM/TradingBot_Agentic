import { expect, test, type Page } from '@playwright/test';

/**
 * SC-2e — the mode toggle that makes the conversational scanner discoverable.
 *
 * Before this, `/scanner/decrire` existed but nothing linked to it. The toggle
 * sits at the top of BOTH scanner surfaces: the active mode is highlighted and
 * inert (`aria-current`), the other is a locale-preserving link.
 *
 * We assert the load-bearing contract deterministically: on each surface the
 * toggle is present, the current mode is marked, and the OTHER mode is a link
 * whose href targets the other surface (locale preserved — the default locale
 * `fr` is served prefix-less, every other locale under `/<code>`). We also
 * `goto` each target to prove it is reachable and that the toggle stays
 * consistent there. (We assert hrefs rather than driving a click: `next/link`
 * client navigation is timing-flaky under `next dev`; the href IS the
 * discoverability guarantee, and CI serves the production build.)
 *
 * Runs at both viewports via the Playwright projects (desktop + mobile).
 */

async function mockAccess(page: Page) {
  await page.route('**/api/access/me', (r) =>
    r.fulfill({ json: { authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false, is_owner: true, has_access: true, subscription_required: false } }),
  );
  await page.route('**/api/auth/me', (r) => r.fulfill({ json: { account: null } }));
  // The classic scanner auto-scans; keep it deterministic and offline.
  await page.route('**/api/conditions-scan', (r) =>
    r.fulfill({ json: { generated_at: new Date().toISOString(), results: [] } }),
  );
}

test.describe('SC-2e — scanner mode toggle', () => {
  test('classic scanner: toggle present, « Décrire » links to the conversational mode', async ({
    page,
  }) => {
    await mockAccess(page);
    await page.goto('/scanner');

    const group = page.getByRole('group', { name: 'Mode du scanner' });
    await expect(group).toBeVisible();
    await expect(group.getByText('Choisir mes conditions')).toHaveAttribute('aria-current', 'page');

    const describeLink = group.getByRole('link', { name: 'Décrire ma stratégie' });
    await expect(describeLink).toHaveAttribute('href', '/scanner/decrire');
  });

  test('conversational scanner: toggle present, « Choisir mes conditions » links back', async ({
    page,
  }) => {
    await mockAccess(page);
    await page.goto('/scanner/decrire');

    const group = page.getByRole('group', { name: 'Mode du scanner' });
    await expect(group).toBeVisible();
    await expect(group.getByText('Décrire ma stratégie')).toHaveAttribute('aria-current', 'page');

    const conditionsLink = group.getByRole('link', { name: 'Choisir mes conditions' });
    await expect(conditionsLink).toHaveAttribute('href', '/scanner');
  });

  test('the two surfaces are reachable and the toggle stays consistent', async ({ page }) => {
    await mockAccess(page);
    // Target of the classic → describe link renders the conversational scanner.
    await page.goto('/scanner/decrire');
    await expect(page.getByTestId('describe-input')).toBeVisible();
    // Target of the describe → conditions link renders the classic scanner + toggle.
    await page.goto('/scanner');
    await expect(page.getByRole('group', { name: 'Mode du scanner' })).toBeVisible();
  });

  test('the toggle preserves a non-default locale (en → /en/scanner/decrire)', async ({ page }) => {
    await mockAccess(page);
    await page.goto('/en/scanner');

    const group = page.getByRole('group', { name: 'Scanner mode' });
    await expect(group).toBeVisible();
    await expect(group.getByText('Choose my conditions')).toHaveAttribute('aria-current', 'page');
    await expect(group.getByRole('link', { name: 'Describe my strategy' })).toHaveAttribute(
      'href',
      '/en/scanner/decrire',
    );
  });
});
