import { expect, test } from '@playwright/test';

/**
 * Sections collapsibles de la MarketReadingCard (Chantier 5.B) telles que
 * rendues dans la galerie multi-marché de la landing. Mises à jour au
 * Chantier 5.C : la landing consomme désormais la card market-reading native
 * (Structure / Régime / Événements / Lecture narrée) — plus de section
 * « Détail expert », plus de waterfall de score, plus de champ BOCPD.
 */
test.describe('Sections collapsibles — golden paths', () => {
  test('clicking the structure trigger reveals SMC facts', async ({ page }) => {
    await page.goto('/#multi-marche');
    // First card's Structure trigger (XAU M15).
    const structureTrigger = page
      .getByRole('button', { name: /Structure de marché/i })
      .first();
    await structureTrigger.click();
    // BOS line of the XAU fixture (price · direction · validation).
    await expect(page.getByText(/haussier · confirmée/i).first()).toBeVisible();
  });

  test('clicking the regime trigger reveals the MTF confluence map', async ({ page }) => {
    await page.goto('/#multi-marche');
    const regimeTrigger = page
      .getByRole('button', { name: /Régime de marché/i })
      .first();
    await regimeTrigger.click();
    await expect(
      page.getByText(/Confluence multi-timeframe/i).first(),
    ).toBeVisible();
  });

  test('narrated reading section shows the plain-language conditions on click', async ({ page }) => {
    await page.goto('/#multi-marche');
    const conditionsTrigger = page
      .getByRole('button', { name: /Lecture narrée/i })
      .first();
    await expect(conditionsTrigger).toBeVisible();
    await conditionsTrigger.click();
    await expect(
      page.getByText(/Structure haussière confirmée/i).first(),
    ).toBeVisible();
  });
});
