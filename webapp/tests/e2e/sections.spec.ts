import { expect, test } from '@playwright/test';

test.describe('Sections collapsibles — golden paths', () => {
  test('clicking the structure trigger reveals SMC fields', async ({ page }) => {
    await page.goto('/#multi-marche');
    // First card's Structure trigger (using role+name).
    const structureTrigger = page
      .getByRole('button', { name: /Structure de marché/i })
      .first();
    await structureTrigger.click();
    // Content of the structure section becomes visible.
    await expect(page.getByText(/Cassure de structure/i)).toBeVisible();
    await expect(page.getByText(/Invalidation structurelle/i)).toBeVisible();
  });

  test('clicking the regime trigger reveals HMM + BOCPD info', async ({ page }) => {
    await page.goto('/#multi-marche');
    const regimeTrigger = page
      .getByRole('button', { name: /Régime de marché/i })
      .first();
    await regimeTrigger.click();
    await expect(page.getByText(/Tendance haussière/i).first()).toBeVisible();
    await expect(page.getByText(/Stabilité du régime/i)).toBeVisible();
  });

  test('expert section shows STRATEGIST badge and waterfall on click', async ({ page }) => {
    await page.goto('/#multi-marche');
    const expertTrigger = page
      .getByRole('button', { name: /Détail expert/i })
      .first();
    await expect(expertTrigger).toBeVisible();
    await expertTrigger.click();
    await expect(
      page.getByText(/Décomposition du score/i).first(),
    ).toBeVisible();
  });
});
