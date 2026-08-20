import { expect, test, type Page } from '@playwright/test';

/**
 * CF-1 — At first load every condition family loads COLLAPSED.
 *
 * The four families (01 Structure, 02 Zones, 03 Liquidité, 04 Contexte) must each
 * render with title + active-count + chevron visible but content folded. No family
 * opens automatically; the client expands what they want via the chevron. A family
 * that holds active conditions still shows collapsed but keeps its count and its
 * selections. Verified at both viewports.
 */

async function mock(page: Page) {
  await page.route('**/api/access/me', (r) =>
    r.fulfill({ json: { authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false, is_owner: true, has_access: true, subscription_required: false } }),
  );
  await page.route('**/api/conditions-scan', (r) =>
    r.fulfill({ json: { as_of: new Date().toISOString(), logic: 'AND', scanned: 0, matches: [], unavailable: [] } }),
  );
}

const FAMILIES = ['Structure', 'Zones', 'Liquidité', 'Contexte'];

for (const vp of [
  { name: 'desktop 1280×800', width: 1280, height: 800 },
  { name: 'phone 390×844', width: 390, height: 844 },
]) {
  test.describe(vp.name, () => {
    test.beforeEach(async ({ page }) => {
      await page.setViewportSize({ width: vp.width, height: vp.height });
    });

    test('all four families load collapsed, no content shown', async ({ page }) => {
      await mock(page);
      await page.goto('/scanner');

      // Every family header is present and folded (aria-expanded=false).
      for (const name of FAMILIES) {
        const header = page.getByRole('button', { name: new RegExp(name) }).first();
        await expect(header).toBeVisible();
        await expect(header).toHaveAttribute('aria-expanded', 'false');
      }
      // Folded → no condition checkbox is rendered yet.
      await expect(page.getByRole('checkbox')).toHaveCount(0);

      await page.screenshot({ path: `test-results/cf1-collapsed-${vp.width}.png`, fullPage: true });
    });

    test('folding a family keeps its active count and its selections', async ({ page }) => {
      await mock(page);
      await page.goto('/scanner');

      const structure = page.getByRole('button', { name: /Structure/ }).first();
      // Expand Structure and tick its first two conditions.
      await structure.click();
      const boxes = page.getByRole('checkbox');
      await boxes.nth(0).check();
      await boxes.nth(1).check();
      await expect(structure.getByText('2 actives')).toBeVisible();

      // Fold it back: the count survives, the ticks are NOT cleared.
      await structure.click();
      await expect(structure).toHaveAttribute('aria-expanded', 'false');
      await expect(structure.getByText('2 actives')).toBeVisible();

      // Re-expand: the two conditions are still checked.
      await structure.click();
      await expect(page.getByRole('checkbox').nth(0)).toBeChecked();
      await expect(page.getByRole('checkbox').nth(1)).toBeChecked();
    });

    test('clicking a chevron expands that family only, others stay folded', async ({ page }) => {
      await mock(page);
      await page.goto('/scanner');

      const structure = page.getByRole('button', { name: /Structure/ }).first();
      const zones = page.getByRole('button', { name: /Zones/ }).first();

      await structure.click();
      await expect(structure).toHaveAttribute('aria-expanded', 'true');
      await expect(zones).toHaveAttribute('aria-expanded', 'false');
      await expect(page.getByRole('checkbox').first()).toBeVisible();

      // Multiple families may be open at once — opening Zones does not close Structure.
      await zones.click();
      await expect(structure).toHaveAttribute('aria-expanded', 'true');
      await expect(zones).toHaveAttribute('aria-expanded', 'true');
    });
  });
}
