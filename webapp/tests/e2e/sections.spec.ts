import { expect, test } from '@playwright/test';
import { dismissCookieBanner } from './utils';

/**
 * Sections collapsibles de la MarketReadingCard (Chantier 5.B) telles que
 * rendues dans la galerie multi-marché de la landing. Mises à jour au
 * Chantier 5.C : la landing consomme désormais la card market-reading native
 * (Structure / Régime / Événements / Lecture narrée) — plus de section
 * « Détail expert », plus de waterfall de score, plus de champ BOCPD.
 */
test.describe('Sections collapsibles — golden paths', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/#multi-marche');
    // Mobile : la bannière cookies (fixed bottom, z-50) intercepte les clics
    // sur les triggers d'accordéon proches du bord inférieur.
    await dismissCookieBanner(page);
  });

  test('clicking the structure trigger reveals SMC facts', async ({ page }) => {
    // First card's Structure trigger (XAU M15).
    const structureTrigger = page
      .getByRole('button', { name: /Structure de marché/i })
      .first();
    await structureTrigger.click();
    // BOS line of the XAU fixture (price · direction · validation).
    await expect(page.getByText(/haussier · confirmée/i).first()).toBeVisible();
  });

  test('clicking the regime trigger reveals the MTF alignment', async ({ page }) => {
    const regimeTrigger = page
      .getByRole('button', { name: /Régime de marché/i })
      .first();
    await regimeTrigger.click();
    // Libellé actuel de la RegimeSection (enrichissement 2026-06-29) — le
    // paragraphe s'affiche aussi sans backend (« … indisponible »).
    await expect(
      page.getByText(/Alignement multi-timeframe/i).first(),
    ).toBeVisible();
  });

  test('narrated reading section shows the plain-language conditions on click', async ({ page }) => {
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
