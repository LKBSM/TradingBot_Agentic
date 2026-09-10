import { expect, test, type Page } from '@playwright/test';
import { dismissCookieBanner } from './utils';
import frMessages from '@/messages/fr.json';
import enMessages from '@/messages/en.json';

/**
 * The visitor CTA label is READ from the messages, never hardcoded: PAY-2
 * renamed it « Essayer gratuitement » → « S'abonner » and this spec silently
 * rotted for a whole release. What the test guards is the RULE — a logged-out
 * visitor gets the sign-up CTA and no App/Zones/Scanner — not the wording.
 */
function ctaLabel(messages: { nav: { tryFree: string } }): RegExp {
  return new RegExp(messages.nav.tryFree.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i');
}

/**
 * LP-1 — home page. Runs fr (/) and en (/en) at desktop 1280×800 and mobile
 * 390×844, exercising the full page, each interactive demo in two states, and
 * the pricing block. The page is fully static (illustration data) — no network.
 */

type Loc = {
  code: string;
  path: string;
  h1: RegExp;
  chipStr: string;
  chipOb: string;
  chipFvg: string;
  chipLiq: string;
  chochFrag: RegExp;
  liqFrag: RegExp;
  emptyLayers: RegExp;
  scannerTab: RegExp;
  trend: string;
  higher: string;
  ob: string;
  untested: string;
  swept: string;
  noCond: RegExp;
  noMatch: RegExp;
  structureTab: RegExp;
  fvgFrag: RegExp;
  miaTab: RegExp;
  miaAction: string;
  miaChanged: RegExp;
  miaLiveAnswer: string;
  miaOffline: RegExp;
  miaPlaceholder: string;
  miaSend: string;
  miaTyped: string;
  calcTab: RegExp;
  calcVerdict: string;
  calcOpen: string;
  calcRow: string;
  illus: RegExp;
  marketsLine: RegExp;
  marketsTickers: RegExp;
  tryFree: RegExp;
  dot5: RegExp;
};

const LOCALES: Loc[] = [
  {
    code: 'fr',
    path: '/',
    h1: /MIA te le lit/i,
    chipStr: 'BOS / CHOCH',
    chipOb: 'Order Blocks',
    chipFvg: 'Fair Value Gaps',
    chipLiq: 'Liquidité',
    chochFrag: /CHOCH haussier/i,
    liqFrag: /liquidité achat reste intacte/i,
    emptyLayers: /n'invente rien pour remplir le vide/i,
    scannerTab: /Définir une stratégie/i,
    trend: 'La tendance structurelle est haussière',
    higher: "L'unité supérieure va dans le même sens",
    ob: 'Le prix est dans un Order Block',
    untested: "La zone n'a jamais été testée",
    swept: 'Une poche a été prise récemment',
    noCond: /et surtout pas tous les marchés/i,
    noMatch: /Ce n'est pas une erreur/i,
    structureTab: /Lire une structure/i,
    fvgFrag: /Fair Value Gap baissier comblé/i,
    miaTab: /Parler à M\.I\.A/i,
    miaAction: 'Montre-moi seulement les OB non testés',
    miaChanged: /Les couches du graphique ont changé/i,
    miaLiveAnswer: 'Réponse en direct de la démonstration.',
    miaOffline: /La démonstration en direct n'est pas disponible ici/i,
    miaPlaceholder: 'Pose ta question…',
    miaSend: 'Envoyer',
    miaTyped: "Combien coûte l'abonnement ?",
    calcTab: /Ouvrir le calcul/i,
    calcVerdict: 'Normale',
    calcOpen: 'Ouvre le calcul',
    calcRow: 'Parcours moyen récent',
    illus: /Données d'illustration/i,
    marketsLine: /80 marchés au programme/i,
    marketsTickers: /XAUUSD et EURUSD/i,
    tryFree: ctaLabel(frMessages),
    dot5: /Aller au volet 5/i,
  },
  {
    code: 'en',
    path: '/en',
    h1: /MIA reads it to you/i,
    chipStr: 'BOS / CHOCH',
    chipOb: 'Order Blocks',
    chipFvg: 'Fair Value Gaps',
    chipLiq: 'Liquidity',
    chochFrag: /bullish CHOCH confirmed/i,
    liqFrag: /buy-side liquidity pocket stays intact/i,
    emptyLayers: /invents nothing to fill the void/i,
    scannerTab: /Define a strategy/i,
    trend: 'The structural trend is bullish',
    higher: 'The higher timeframe agrees',
    ob: 'Price is inside an Order Block',
    untested: 'The zone has never been tested',
    swept: 'A pocket was taken recently',
    noCond: /and above all not every market/i,
    noMatch: /This is not an error/i,
    structureTab: /Read a structure/i,
    fvgFrag: /bearish Fair Value Gap, 60 % filled/i,
    miaTab: /Talk to M\.I\.A/i,
    miaAction: 'Show me only the untested OBs',
    miaChanged: /The chart layers changed/i,
    miaLiveAnswer: 'Live answer from the demo.',
    miaOffline: /The live demo isn't available here/i,
    miaPlaceholder: 'Ask your question…',
    miaSend: 'Send',
    miaTyped: 'How much is the subscription?',
    calcTab: /Open the calculation/i,
    calcVerdict: 'Normal',
    calcOpen: 'Open the calculation',
    calcRow: 'Recent average range',
    illus: /Illustration data/i,
    marketsLine: /80 markets planned/i,
    marketsTickers: /XAUUSD and EURUSD/i,
    tryFree: ctaLabel(enMessages),
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

      test('full page: hero, honest markets line, illustration mention, pricing, legal', async ({ page }) => {
        await open(page, loc);
        await expect(page.getByRole('heading', { level: 1 })).toContainText(loc.h1);
        // LP-2S — the four-stat banner is gone; one line states the ambition WITH
        // its scope word, then the perimeter actually live. Both halves visible.
        await expect(page.getByText(loc.marketsLine).first()).toBeVisible();
        await expect(page.getByText(loc.marketsTickers).first()).toBeVisible();
        // the removed tiles must not come back, and the maquette fiction stays out
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
        // state B: untick the LAYER CHIPS down to liquidity alone — the chips
        // are the control the side paragraph points at ("untick a layer").
        await demo.getByRole('button', { name: loc.chipStr }).click();
        await expect(demo.getByText(loc.chochFrag)).toHaveCount(0);
        await demo.getByRole('button', { name: loc.chipOb }).click();
        await demo.getByRole('button', { name: loc.chipFvg }).click();
        await expect(demo.getByText(loc.liqFrag).first()).toBeVisible();
        // state C: nothing left → the honest empty state, not an invented filler
        await demo.getByRole('button', { name: loc.chipLiq }).click();
        await expect(demo.getByText(loc.emptyLayers).first()).toBeVisible();
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

      // MIA-4S — the M.I.A tab is the one demo that calls a backend (the real
      // agent, on the frozen scenario). Both halves are pinned: the live answer
      // when the endpoint replies, and the honest degradation when it does not.
      test('demo 4 — M.I.A answers live and can move the chart layers', async ({ page }) => {
        await page.route('**/api/demo/chat/stream', (route) =>
          route.fulfill({
            status: 200,
            contentType: 'text/event-stream',
            body:
              `data: ${JSON.stringify({ event: 'activity' })}\n\n` +
              `data: ${JSON.stringify({
                event: 'answer',
                content: loc.miaLiveAnswer,
                blocked_reason: null,
                tool_calls_made: [],
                view_actions: [
                  { action: 'set_layer_visibility', params: { layer: 'fvg', visible: false } },
                ],
                messages_left: 5,
              })}\n\n`,
          }),
        );
        await open(page, loc);
        const demo = page.locator('#demo');
        await demo.getByRole('tab', { name: loc.miaTab }).click();
        await demo.getByRole('button', { name: loc.miaAction }).click();
        await expect(demo.getByText(loc.miaLiveAnswer).first()).toBeVisible();
        await expect(demo.getByText(loc.miaChanged).first()).toBeVisible();
        // the validated action really reached the structure narration
        await demo.getByRole('tab', { name: loc.structureTab }).click();
        await expect(demo.getByText(loc.fvgFrag)).toHaveCount(0);
      });

      test('demo 4 — with no backend, M.I.A degrades to recorded exchanges and says so', async ({ page }) => {
        await page.route('**/api/demo/chat/stream', (route) => route.abort());
        await open(page, loc);
        const demo = page.locator('#demo');
        await demo.getByRole('tab', { name: loc.miaTab }).click();
        await demo.getByRole('button', { name: loc.miaAction }).click();
        await expect(demo.getByText(loc.miaOffline).first()).toBeVisible();
      });

      test('demo 4 — any question can be typed, the starters are not a menu', async ({ page }) => {
        await page.route('**/api/demo/chat/stream', (route) =>
          route.fulfill({
            status: 200,
            contentType: 'text/event-stream',
            body: `data: ${JSON.stringify({
              event: 'answer',
              content: loc.miaLiveAnswer,
              messages_left: 4,
              view_actions: [],
            })}\n\n`,
          }),
        );
        await open(page, loc);
        const demo = page.locator('#demo');
        await demo.getByRole('tab', { name: loc.miaTab }).click();
        await demo.getByLabel(loc.miaPlaceholder).fill(loc.miaTyped);
        await demo.getByRole('button', { name: loc.miaSend }).click();
        await expect(demo.getByText(loc.miaLiveAnswer).first()).toBeVisible();
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
        test('nav bar: a visitor gets no App/Zones/Scanner, sees the sign-up CTA', async ({ page }) => {
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
