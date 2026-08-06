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
  statStructures: RegExp;
  tryFree: RegExp;
  dot5: RegExp;
};

const LOCALES: Loc[] = [
  {
    code: 'fr',
    path: '/',
    h1: /MIA te le lit/i,
    onlyLiq: 'Ne garder que la liquidité',
    chochFrag: /CHOCH haussier/i,
    liqFrag: /liquidité achat reste intacte/i,
    scannerTab: /Définir une stratégie/i,
    trend: 'La tendance structurelle est haussière',
    higher: "L'unité supérieure va dans le même sens",
    ob: 'Le prix est dans un Order Block',
    untested: "La zone n'a jamais été testée",
    swept: 'Une poche a été prise récemment',
    noCond: /et surtout pas tous les marchés/i,
    noMatch: /Ce n'est pas une erreur/i,
    miaTab: /Parler à M\.I\.A/i,
    miaAction: 'Montre-moi seulement les OB non testés',
    miaChanged: /Les couches du graphique ont changé/i,
    calcTab: /Ouvrir le calcul/i,
    calcVerdict: 'Normale',
    calcOpen: 'Ouvre le calcul',
    calcRow: 'Parcours moyen récent',
    illus: /Données d'illustration/i,
    statStructures: /structures détectées/i,
    tryFree: /Essayer gratuitement/i,
    dot5: /Aller au volet 5/i,
  },
  {
    code: 'en',
    path: '/en',
    h1: /MIA reads it to you/i,
    onlyLiq: 'Keep only the liquidity',
    chochFrag: /bullish CHOCH confirmed/i,
    liqFrag: /buy-side liquidity pocket stays intact/i,
    scannerTab: /Define a strategy/i,
    trend: 'The structural trend is bullish',
    higher: 'The higher timeframe agrees',
    ob: 'Price is inside an Order Block',
    untested: 'The zone has never been tested',
    swept: 'A pocket was taken recently',
    noCond: /and above all not every market/i,
    noMatch: /This is not an error/i,
    miaTab: /Talk to M\.I\.A/i,
    miaAction: 'Show me only the untested OBs',
    miaChanged: /The chart layers changed/i,
    calcTab: /Open the calculation/i,
    calcVerdict: 'Normal',
    calcOpen: 'Open the calculation',
    calcRow: 'Recent average range',
    illus: /Illustration data/i,
    statStructures: /structures detected/i,
    tryFree: /Try for free/i,
    dot5: /Go to panel 5/i,
  },
];

const VIEWPORTS = [
  { name: 'desktop', width: 1280, height: 800 },
  { name: 'mobile', width: 390, height: 844 },
];

async function open(page: Page, loc: Loc) {
  // The page uses `scroll-behavior: smooth`; emulate reduced motion so
  // Playwright's scroll-into-view is instant and click targets stay stable.
  await page.emulateMedia({ reducedMotion: 'reduce' });
  // Tolerate a cold `next dev` first-compile (CI serves the prebuilt `next
  // start`, which is faster); the page itself is static once compiled.
  await page.goto(loc.path, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await dismissCookieBanner(page);
  // Hard-kill smooth scroll and animations for deterministic scroll-into-view:
  // on this long page a mid-flight smooth scroll makes click targets "unstable".
  await page.addStyleTag({
    content:
      '*,*::before,*::after{scroll-behavior:auto !important;transition-duration:0s !important;animation-duration:0s !important}',
  });
}


for (const loc of LOCALES) {
  for (const vp of VIEWPORTS) {
    test.describe(`LP-2 accueil · ${loc.code} · ${vp.name}`, () => {
      // Reduced motion at context creation makes scroll-into-view instant so
      // click targets stay stable on this long, smooth-scrolling page. A couple
      // of retries absorb the residual scroll-container timing flake.
      test.use({
        viewport: { width: vp.width, height: vp.height },
        contextOptions: { reducedMotion: 'reduce' },
      });
      test.describe.configure({ retries: 2 });

      test('full page: hero, real stats, illustration mention, pricing, legal', async ({ page }) => {
        await open(page, loc);
        await expect(page.getByRole('heading', { level: 1 })).toContainText(loc.h1);
        // real figures, not the maquette fictions — LP-2 band ends on "structures"
        await expect(page.getByText('22', { exact: true }).first()).toBeVisible();
        await expect(page.getByText(loc.statStructures).first()).toBeVisible();
        await expect(page.locator('body')).not.toContainText('480');
        // illustration mention present at least once
        await expect(page.getByText(loc.illus).first()).toBeVisible();
        // pricing shows currency everywhere
        await expect(page.getByText('39 $').first()).toBeVisible();
        await expect(page.getByText(/348 \$ US/).first()).toBeVisible();
      });

      test('reading-space carousel: keyboard + dots navigate panels', async ({ page }) => {
        await open(page, loc);
        const region = page.locator('[aria-roledescription="carousel"]');
        await region.scrollIntoViewIfNeeded();
        await expect(region.getByText(/1 \/ 5/)).toBeVisible();
        // keyboard: two panels forward
        await region.focus();
        await page.keyboard.press('ArrowRight');
        await expect(region.getByText(/2 \/ 5/)).toBeVisible();
        await page.keyboard.press('ArrowRight');
        await expect(region.getByText(/3 \/ 5/)).toBeVisible();
        // dots: jump to the last panel
        await region.getByRole('button', { name: loc.dot5 }).click();
        await expect(region.getByText(/5 \/ 5/)).toBeVisible();
      });

      // The interactive demos are scoped to the #demo section: the reading-space
      // carousel below reuses some of the same narrated phrases (illustration
      // data), so a page-wide query would double-count them.
      test('demo 1 — structure narration rewrites (two states)', async ({ page }) => {
        await open(page, loc);
        const demo = page.locator('#demo');
        // state A: all layers → CHOCH in narration
        await expect(demo.getByText(loc.chochFrag).first()).toBeVisible();
        // state B: keep only liquidity → CHOCH gone, liquidity present
        await demo.getByRole('button', { name: loc.onlyLiq }).click();
        await expect(demo.getByText(loc.chochFrag)).toHaveCount(0);
        await expect(demo.getByText(loc.liqFrag).first()).toBeVisible();
      });

      test('demo 2 — scanner honest empty states (two states)', async ({ page }) => {
        await open(page, loc);
        const demo = page.locator('#demo');
        await demo.getByRole('tab', { name: loc.scannerTab }).click();
        // state A: no condition → not "all markets" (buttons carry a ✓ prefix
        // when checked, so match by role name rather than exact text)
        await demo.getByRole('button', { name: loc.trend }).click();
        await demo.getByRole('button', { name: loc.higher }).click();
        await expect(demo.getByText(loc.noCond).first()).toBeVisible();
        // state B: restrictive combo → "not an error"
        await demo.getByRole('button', { name: loc.ob }).click();
        await demo.getByRole('button', { name: loc.untested }).click();
        await demo.getByRole('button', { name: loc.swept }).click();
        await expect(demo.getByText(loc.noMatch).first()).toBeVisible();
      });

      test('demo 4 — a MIA question changes the chart layers', async ({ page }) => {
        await open(page, loc);
        const demo = page.locator('#demo');
        await demo.getByRole('tab', { name: loc.miaTab }).click();
        await demo.getByRole('button', { name: loc.miaAction }).click();
        await expect(demo.getByText(loc.miaChanged).first()).toBeVisible();
      });

      test('demo 5 — régime tile reveals the raw calculation', async ({ page }) => {
        await open(page, loc);
        const demo = page.locator('#demo');
        await demo.getByRole('tab', { name: loc.calcTab }).click();
        // confirm the pane actually switched before asserting on it
        await expect(demo.getByText(loc.calcVerdict, { exact: true })).toBeVisible();
        await expect(demo.getByText(loc.calcRow)).toHaveCount(0);
        await demo.getByRole('button', { name: loc.calcOpen }).click();
        await expect(demo.getByText(loc.calcRow).first()).toBeVisible();
      });

      if (vp.name === 'desktop') {
        test('nav bar: a visitor gets no App/Zones/Scanner, sees the free-trial CTA', async ({ page }) => {
          await open(page, loc);
          const header = page.locator('header').first();
          // gate the assertions on the resolved logged-out state
          await expect(header.getByRole('link', { name: loc.tryFree })).toBeVisible();
          await expect(header.getByRole('link', { name: /^Zones$/ })).toHaveCount(0);
          await expect(header.getByRole('link', { name: /^Scanner$/ })).toHaveCount(0);
          await expect(header.getByRole('link', { name: /^App$/ })).toHaveCount(0);
        });

        test('nav bar: an authenticated visitor gets the product links', async ({ page }) => {
          // Mock the session probe so the nav renders its logged-in cluster.
          await page.route('**/api/auth/me', (route) =>
            route.fulfill({
              status: 200,
              contentType: 'application/json',
              body: JSON.stringify({ id: 'u1', email: 'test@example.com', tier: 'institutional' }),
            }),
          );
          await open(page, loc);
          const header = page.locator('header').first();
          await expect(header.getByRole('link', { name: /^Zones$/ }).first()).toBeVisible();
          await expect(header.getByRole('link', { name: /^Scanner$/ }).first()).toBeVisible();
        });
      }
    });
  }
}
