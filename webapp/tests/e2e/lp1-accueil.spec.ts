import { expect, test, type Page } from '@playwright/test';
import { dismissCookieBanner } from './utils';

/**
 * LP-1 — home page. Runs fr (/) and en (/en) at desktop 1280×800 and mobile
 * 390×844, exercising the full page, each interactive demo in two states, and
 * the pricing block. The page is fully static (illustration data) — no network.
 */

type Loc = {
  code: string;
  path: string;
  h1: RegExp;
  onlyLiq: string;
  chochFrag: RegExp;
  liqFrag: RegExp;
  scannerTab: RegExp;
  trend: string;
  higher: string;
  ob: string;
  untested: string;
  swept: string;
  noCond: RegExp;
  noMatch: RegExp;
  miaTab: RegExp;
  miaAction: string;
  miaChanged: RegExp;
  calcTab: RegExp;
  calcVerdict: string;
  calcOpen: string;
  calcRow: string;
  illus: RegExp;
};

const LOCALES: Loc[] = [
  {
    code: 'fr',
    path: '/',
    h1: /MIA te le lit/i,
    onlyLiq: 'Ne garder que la liquidité',
    chochFrag: /CHOCH haussier/i,
    liqFrag: /liquidité achat reste intacte/i,
    scannerTab: /Chercher un marché/i,
    trend: 'La tendance structurelle est haussière',
    higher: "L'unité supérieure va dans le même sens",
    ob: 'Le prix est dans un Order Block',
    untested: "La zone n'a jamais été testée",
    swept: 'Une poche a été prise récemment',
    noCond: /et surtout pas tous les marchés/i,
    noMatch: /Ce n'est pas une erreur/i,
    miaTab: /Poser une question/i,
    miaAction: 'Montre-moi seulement les OB non testés',
    miaChanged: /Les couches du graphique ont changé/i,
    calcTab: /Ouvrir le calcul/i,
    calcVerdict: 'Normale',
    calcOpen: 'Ouvre le calcul',
    calcRow: 'Parcours moyen récent',
    illus: /Données d'illustration/i,
  },
  {
    code: 'en',
    path: '/en',
    h1: /MIA reads it to you/i,
    onlyLiq: 'Keep only the liquidity',
    chochFrag: /bullish CHOCH confirmed/i,
    liqFrag: /buy-side liquidity pocket stays intact/i,
    scannerTab: /Search a market/i,
    trend: 'The structural trend is bullish',
    higher: 'The higher timeframe agrees',
    ob: 'Price is inside an Order Block',
    untested: 'The zone has never been tested',
    swept: 'A pocket was taken recently',
    noCond: /and above all not every market/i,
    noMatch: /This is not an error/i,
    miaTab: /Ask a question/i,
    miaAction: 'Show me only the untested OBs',
    miaChanged: /The chart layers changed/i,
    calcTab: /Open the calculation/i,
    calcVerdict: 'Normal',
    calcOpen: 'Open the calculation',
    calcRow: 'Recent average range',
    illus: /Illustration data/i,
  },
];

const VIEWPORTS = [
  { name: 'desktop', width: 1280, height: 800 },
  { name: 'mobile', width: 390, height: 844 },
];

async function open(page: Page, loc: Loc) {
  // Tolerate a cold `next dev` first-compile (CI serves the prebuilt `next
  // start`, which is faster); the page itself is static once compiled.
  await page.goto(loc.path, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await dismissCookieBanner(page);
}

for (const loc of LOCALES) {
  for (const vp of VIEWPORTS) {
    test.describe(`LP-1 accueil · ${loc.code} · ${vp.name}`, () => {
      test.use({ viewport: { width: vp.width, height: vp.height } });

      test('full page: hero, real stats, illustration mention, pricing, legal', async ({ page }) => {
        await open(page, loc);
        await expect(page.getByRole('heading', { level: 1 })).toContainText(loc.h1);
        // real figures, not the maquette fictions
        await expect(page.getByText('12', { exact: true }).first()).toBeVisible();
        await expect(page.getByText('22', { exact: true }).first()).toBeVisible();
        await expect(page.locator('body')).not.toContainText('480');
        // illustration mention present at least once
        await expect(page.getByText(loc.illus).first()).toBeVisible();
        // pricing shows currency everywhere
        await expect(page.getByText('39 $').first()).toBeVisible();
        await expect(page.getByText(/348 \$ US/).first()).toBeVisible();
      });

      test('demo 1 — structure narration rewrites (two states)', async ({ page }) => {
        await open(page, loc);
        // state A: all layers → CHOCH in narration
        await expect(page.getByText(loc.chochFrag).first()).toBeVisible();
        // state B: keep only liquidity → CHOCH gone, liquidity present
        await page.getByRole('button', { name: loc.onlyLiq }).click();
        await expect(page.getByText(loc.chochFrag)).toHaveCount(0);
        await expect(page.getByText(loc.liqFrag).first()).toBeVisible();
      });

      test('demo 2 — scanner honest empty states (two states)', async ({ page }) => {
        await open(page, loc);
        await page.getByRole('tab', { name: loc.scannerTab }).click();
        // state A: no condition → not "all markets" (buttons carry a ✓ prefix
        // when checked, so match by role name rather than exact text)
        await page.getByRole('button', { name: loc.trend }).click();
        await page.getByRole('button', { name: loc.higher }).click();
        await expect(page.getByText(loc.noCond).first()).toBeVisible();
        // state B: restrictive combo → "not an error"
        await page.getByRole('button', { name: loc.ob }).click();
        await page.getByRole('button', { name: loc.untested }).click();
        await page.getByRole('button', { name: loc.swept }).click();
        await expect(page.getByText(loc.noMatch).first()).toBeVisible();
      });

      test('demo 4 — a MIA question changes the chart layers', async ({ page }) => {
        await open(page, loc);
        await page.getByRole('tab', { name: loc.miaTab }).click();
        await page.getByRole('button', { name: loc.miaAction }).click();
        await expect(page.getByText(loc.miaChanged).first()).toBeVisible();
      });

      test('demo 5 — régime tile reveals the raw calculation', async ({ page }) => {
        await open(page, loc);
        await page.getByRole('tab', { name: loc.calcTab }).click();
        // confirm the pane actually switched before asserting on it
        await expect(page.getByText(loc.calcVerdict, { exact: true })).toBeVisible();
        await expect(page.getByText(loc.calcRow)).toHaveCount(0);
        await page.getByRole('button', { name: loc.calcOpen }).click();
        await expect(page.getByText(loc.calcRow).first()).toBeVisible();
      });
    });
  }
}
