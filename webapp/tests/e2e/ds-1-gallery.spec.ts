import { test, expect } from '@playwright/test';
import path from 'node:path';

/**
 * DS-1 — the design gallery renders with NO backend running (only `next dev`,
 * no FastAPI). Every surface is present, fed by the frozen real-data samples.
 * Captured at both target viewports for Claude Design.
 *
 * The gallery is a dev-only route (production 404s), so this runs against the
 * dev server (E2E_BASE_URL) — never a production build.
 */

const OUT = path.resolve(__dirname, '../../../docs/audits/ds-1');
const SURFACES = ['market-reading', 'zones', 'chart', 'scanner', 'mia', 'landing'];

/**
 * L'en-tête ci-dessus le dit déjà : `/galerie` est une route DE DÉVELOPPEMENT,
 * elle 404 en production. Or l'intégration continue sert un build de production
 * (`npm run start`) — ces deux cas y échouaient donc à chaque exécution, depuis
 * toujours, pour une raison qui ne dépend pas du produit : ils éprouvent une
 * surface que l'application servie ne contient pas.
 *
 * On SONDE la route au lieu de déduire l'environnement d'une variable : la
 * précondition réelle est « /galerie répond », pas « CI vaut true ». Un futur
 * job qui servirait `next dev` retrouverait la couverture sans qu'on touche ce
 * fichier.
 *
 * Pour les exécuter :
 *   cd webapp && npm run dev
 *   E2E_BASE_URL=http://localhost:3000 npx playwright test ds-1-gallery \
 *     --project=chromium-desktop --workers=1
 */
async function galerieServie(page: import('@playwright/test').Page): Promise<boolean> {
  try {
    const res = await page.request.get('/galerie', { failOnStatusCode: false });
    return res.status() === 200;
  } catch {
    return false;
  }
}

const RAISON_SAUT =
  'route /galerie absente de cette construction — elle n’existe qu’en ' +
  'développement (la production 404) ; relancer contre `npm run dev` ' +
  '(voir l’en-tête du fichier)';

async function assertGallery(page: import('@playwright/test').Page) {
  // `domcontentloaded` (not networkidle): the app's AuthProvider keeps retrying
  // its session endpoint with no backend, so the network never goes idle.
  await page.goto('/galerie', { waitUntil: 'domcontentloaded' });
  await expect(page.getByTestId('ds-gallery')).toBeVisible();
  for (const s of SURFACES) {
    await expect(page.getByTestId(`ds-surface-${s}`)).toBeVisible();
  }
  // the price-inside-band zone (the gauge edge case) is present
  await expect(page.getByTestId('ds-state-prix-à-l’intérieur')).toBeVisible();
  // let the lightweight-charts canvas paint the candles before the capture
  await page.locator('[data-testid="ds-surface-chart"] canvas').first().waitFor({ state: 'visible', timeout: 15_000 }).catch(() => {});
  await page.waitForTimeout(1_500);
}

test('gallery renders without a backend — desktop 1280×800', async ({ page }) => {
  test.setTimeout(90_000);
  test.skip(!(await galerieServie(page)), RAISON_SAUT);
  await page.setViewportSize({ width: 1280, height: 800 });
  await assertGallery(page);
  await page.screenshot({ path: path.join(OUT, 'gallery-desktop-1280x800.png'), fullPage: true });
});

test('gallery renders without a backend — mobile 390×844', async ({ page }) => {
  test.setTimeout(90_000);
  test.skip(!(await galerieServie(page)), RAISON_SAUT);
  await page.setViewportSize({ width: 390, height: 844 });
  await assertGallery(page);
  await page.screenshot({ path: path.join(OUT, 'gallery-mobile-390x844.png'), fullPage: true });
});
