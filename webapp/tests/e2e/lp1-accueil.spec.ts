import { expect, test, type Page } from '@playwright/test';
import { dismissCookieBanner } from './utils';
import frMessages from '@/messages/fr.json';
import enMessages from '@/messages/en.json';

/**
 * LP-3 — the home page, fr (/) and en (/en), at 1280×800 and 390×844.
 *
 * The spec is in two halves on purpose:
 *
 *   · everything below `for (const loc of LOCALES)` runs with reduced motion,
 *     which is the DEGRADED path of the scroll section: every sentence present,
 *     every layer drawn, chips live. That is the path most assertions want,
 *     because it is stable and because it is what a real visitor with reduced
 *     motion, no JS, or a narrow screen gets. Nothing may be missing from it.
 *
 *   · the last describe deliberately does NOT reduce motion, and is the only
 *     place the pinned choreography is exercised — it needs a real viewport to
 *     scroll, which is exactly what jsdom cannot give home.test.tsx.
 */

/**
 * The visitor CTA label is READ from the messages, never hardcoded: PAY-2
 * renamed it « Essayer gratuitement » → « S'abonner » and this spec silently
 * rotted for a whole release. What the test guards is the RULE — a logged-out
 * visitor gets the sign-up CTA — not the wording.
 */
function ctaLabel(messages: { nav: { tryFree: string } }): RegExp {
  return new RegExp(messages.nav.tryFree.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i');
}

type Loc = {
  code: string;
  path: string;
  h1: RegExp;
  archive: RegExp;
  chipStr: string;
  chipOb: string;
  chipFvg: string;
  chipLiq: string;
  chochFrag: RegExp;
  fvgFrag: RegExp;
  liqFrag: RegExp;
  emptyLayers: RegExp;
  trend: string;
  higher: string;
  ob: string;
  untested: string;
  swept: string;
  noCond: RegExp;
  noMatch: RegExp;
  miaAction: string;
  miaLiveAnswer: string;
  miaOffline: RegExp;
  miaPlaceholder: string;
  miaSend: string;
  miaTyped: string;
  calcOpen: string;
  calcRow: string;
  illus: RegExp;
  marketsLine: RegExp;
  marketsTickers: RegExp;
  tryFree: RegExp;
  stepBreak: RegExp;
  stepLiq: RegExp;
  /** same labels, unanchored — toContainText sees label + sentence. */
  stepBreakText: string;
  stepLiqText: string;
};

const LOCALES: Loc[] = [
  {
    code: 'fr',
    path: '/',
    h1: /Elle lit la structure/i,
    archive: /Lecture réelle, archivée/i,
    chipStr: 'BOS / CHOCH',
    chipOb: 'Order Blocks',
    chipFvg: 'Fair Value Gaps',
    chipLiq: 'Liquidité',
    chochFrag: /CHOCH haussier/i,
    fvgFrag: /Fair Value Gap baissier comblé/i,
    liqFrag: /liquidité achat reste intacte/i,
    emptyLayers: /n'invente rien pour remplir le vide/i,
    trend: 'La tendance structurelle est haussière',
    higher: "L'unité supérieure va dans le même sens",
    ob: 'Le prix est dans un Order Block',
    untested: "La zone n'a jamais été testée",
    swept: 'Une poche a été prise récemment',
    noCond: /et surtout pas tous les marchés/i,
    noMatch: /Ce n'est pas une erreur/i,
    miaAction: 'Montre-moi seulement les OB non testés',
    miaLiveAnswer: 'Réponse en direct de la démonstration.',
    miaOffline: /La démonstration en direct n'est pas disponible ici/i,
    miaPlaceholder: 'Pose ta question…',
    miaSend: 'Envoyer',
    miaTyped: "Combien coûte l'abonnement ?",
    calcOpen: 'Ouvre le calcul',
    calcRow: 'Parcours moyen récent',
    illus: /Données d'illustration/i,
    marketsLine: /80 marchés au programme/i,
    marketsTickers: /XAUUSD et EURUSD/i,
    tryFree: ctaLabel(frMessages),
    stepBreak: /^La cassure$/,
    stepLiq: /^La liquidité$/,
    stepBreakText: 'La cassure',
    stepLiqText: 'La liquidité',
  },
  {
    code: 'en',
    path: '/en',
    h1: /It reads the structure/i,
    archive: /A real, archived reading/i,
    chipStr: 'BOS / CHOCH',
    chipOb: 'Order Blocks',
    chipFvg: 'Fair Value Gaps',
    chipLiq: 'Liquidity',
    chochFrag: /bullish CHOCH confirmed/i,
    fvgFrag: /bearish Fair Value Gap, 60 % filled/i,
    liqFrag: /buy-side liquidity pocket stays intact/i,
    emptyLayers: /invents nothing to fill the void/i,
    trend: 'The structural trend is bullish',
    higher: 'The higher timeframe agrees',
    ob: 'Price is inside an Order Block',
    untested: 'The zone has never been tested',
    swept: 'A pocket was taken recently',
    noCond: /and above all not every market/i,
    noMatch: /This is not an error/i,
    miaAction: 'Show me only the untested OBs',
    miaLiveAnswer: 'Live answer from the demo.',
    miaOffline: /The live demo isn't available here/i,
    miaPlaceholder: 'Ask your question…',
    miaSend: 'Send',
    miaTyped: 'How much is the subscription?',
    calcOpen: 'Open the calculation',
    calcRow: 'Recent average range',
    illus: /Illustration data/i,
    marketsLine: /80 markets planned/i,
    marketsTickers: /XAUUSD and EURUSD/i,
    tryFree: ctaLabel(enMessages),
    stepBreak: /^The break$/,
    stepLiq: /^The liquidity$/,
    stepBreakText: 'The break',
    stepLiqText: 'The liquidity',
  },
];

const VIEWPORTS = [
  { name: 'desktop', width: 1280, height: 800 },
  { name: 'mobile', width: 390, height: 844 },
];

async function open(page: Page, loc: Loc) {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto(loc.path, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await dismissCookieBanner(page);
  await page.addStyleTag({
    content:
      '*,*::before,*::after{scroll-behavior:auto !important;transition-duration:0s !important;animation-duration:0s !important}',
  });
}

for (const loc of LOCALES) {
  for (const vp of VIEWPORTS) {
    test.describe(`LP-3 accueil · ${loc.code} · ${vp.name}`, () => {
      test.use({
        viewport: { width: vp.width, height: vp.height },
        contextOptions: { reducedMotion: 'reduce' },
      });
      test.describe.configure({ retries: 2 });

      test('the opening shows the real product surface and dates it', async ({ page }) => {
        await open(page, loc);
        await expect(page.getByRole('heading', { level: 1 })).toContainText(loc.h1);
        // the archived reading is labelled as archived, never passed off as live
        await expect(page.getByText(loc.archive).first()).toBeVisible();
        // LP-2S: the market perimeter line, ambition qualified
        await expect(page.getByText(loc.marketsLine).first()).toBeVisible();
        await expect(page.getByText(loc.marketsTickers).first()).toBeVisible();
        // the retired stat banner must not come back
        await expect(page.locator('body')).not.toContainText('marchés suivis');
        await expect(page.locator('body')).not.toContainText('markets tracked');
      });

      test('pricing and the legal mentions survive the redesign', async ({ page }) => {
        await open(page, loc);
        await expect(page.getByText(/39 \$/).first()).toBeVisible();
        const body = page.locator('body');
        if (loc.code === 'fr') {
          await expect(body).toContainText('dollars américains');
          await expect(body).toContainText('risque de perte');
          await expect(body).toContainText('18 ans');
        } else {
          await expect(body).toContainText('US dollars');
          await expect(body).toContainText('risk of loss');
        }
        await expect(page.getByText(loc.illus).first()).toBeVisible();
      });

      test('degraded scroll section: every sentence is there, and a chip rewrites it', async ({ page }) => {
        await open(page, loc);
        // reduced motion ⇒ the whole reading, every layer on
        await expect(page.getByText(loc.chochFrag).first()).toBeVisible();
        await expect(page.getByText(loc.fvgFrag).first()).toBeVisible();
        await expect(page.getByText(loc.liqFrag).first()).toBeVisible();
        // the chips are the real control — unticking removes THAT sentence
        await page.getByRole('button', { name: loc.chipStr, exact: true }).click();
        await expect(page.getByText(loc.chochFrag)).toHaveCount(0);
        await expect(page.getByText(loc.liqFrag).first()).toBeVisible();
        // everything off ⇒ the honest empty state
        for (const chip of [loc.chipOb, loc.chipFvg, loc.chipLiq]) {
          await page.getByRole('button', { name: chip, exact: true }).click();
        }
        await expect(page.getByText(loc.emptyLayers).first()).toBeVisible();
      });

      test('the scanner demo is reachable without a click and states both empties', async ({ page }) => {
        await open(page, loc);
        await page.getByRole('button', { name: loc.trend }).click();
        await page.getByRole('button', { name: loc.higher }).click();
        await expect(page.getByText(loc.noCond).first()).toBeVisible();
        await page.getByRole('button', { name: loc.ob }).click();
        await page.getByRole('button', { name: loc.untested }).click();
        await page.getByRole('button', { name: loc.swept }).click();
        await expect(page.getByText(loc.noMatch).first()).toBeVisible();
      });

      test('the régime demo reveals the raw calculation on demand', async ({ page }) => {
        await open(page, loc);
        await expect(page.getByText(loc.calcRow)).toHaveCount(0);
        await page.getByRole('button', { name: loc.calcOpen, exact: true }).click();
        await expect(page.getByText(loc.calcRow).first()).toBeVisible();
      });

      test('M.I.A answers live and her action moves the scroll-section chart', async ({ page }) => {
        await page.route('**/api/demo/chat/stream', async (route) => {
          await route.fulfill({
            status: 200,
            contentType: 'text/event-stream',
            body:
              `data: ${JSON.stringify({ event: 'activity' })}\n\n` +
              `data: ${JSON.stringify({
                event: 'answer',
                content: loc.miaLiveAnswer,
                messages_left: 4,
                view_actions: [
                  { action: 'set_layer_visibility', params: { layer: 'fvg', visible: false } },
                ],
              })}\n\n`,
          });
        });
        await open(page, loc);
        await expect(page.getByText(loc.fvgFrag).first()).toBeVisible();
        await page.getByRole('button', { name: loc.miaAction, exact: true }).click();
        await expect(page.getByText(loc.miaLiveAnswer).first()).toBeVisible();
        // her view action reached the chart at the TOP of the page
        await expect(page.getByText(loc.fvgFrag)).toHaveCount(0);
      });

      test('any question can be typed — the starters are not a menu', async ({ page }) => {
        await page.route('**/api/demo/chat/stream', async (route) => {
          await route.fulfill({
            status: 200,
            contentType: 'text/event-stream',
            body: `data: ${JSON.stringify({ event: 'answer', content: loc.miaLiveAnswer, messages_left: 3 })}\n\n`,
          });
        });
        await open(page, loc);
        await page.getByLabel(loc.miaPlaceholder).fill(loc.miaTyped);
        await page.getByRole('button', { name: loc.miaSend, exact: true }).click();
        await expect(page.getByText(loc.miaLiveAnswer).first()).toBeVisible();
      });

      test('with no backend M.I.A degrades to the recorded exchanges, and says so', async ({ page }) => {
        await page.route('**/api/demo/chat/stream', (route) => route.abort());
        await open(page, loc);
        await page.getByRole('button', { name: loc.miaAction, exact: true }).click();
        await expect(page.getByText(loc.miaOffline).first()).toBeVisible();
      });

      if (vp.name === 'desktop') {
        test('nav bar: a logged-out visitor sees the sign-up CTA', async ({ page }) => {
          await open(page, loc);
          const nav = page.locator('header').first();
          await expect(nav.getByRole('link', { name: loc.tryFree }).first()).toBeVisible();
        });
      }
    });
  }
}

/**
 * The pinned choreography — the ONE orchestrated moment of the page.
 *
 * Deliberately outside the loop above, and deliberately WITHOUT reduced motion:
 * this is the only test that exercises the enhanced path, and it needs a real
 * viewport to scroll. Desktop and fr only; the degraded path (everything else,
 * every locale, both viewports) is covered above.
 */
test.describe('LP-3 — the reading assembles itself as you scroll', () => {
  test.use({ viewport: { width: 1280, height: 900 } });
  test.describe.configure({ retries: 2 });

  test('the first step is alone on the chart, the last one has all four layers', async ({ page }) => {
    const loc = LOCALES[0]!;
    await page.goto(loc.path, { waitUntil: 'domcontentloaded', timeout: 60_000 });
    await dismissCookieBanner(page);

    const section = page.locator('#lecture');
    await section.scrollIntoViewIfNeeded();

    // step 1 — the break is current, and it is the ONLY layer drawn
    await section.getByText(loc.stepBreak).first().scrollIntoViewIfNeeded();
    await expect(section.locator('[aria-current="step"]')).toContainText(loc.stepBreakText);
    await expect(
      section.getByRole('button', { name: loc.chipStr, exact: true }),
    ).toHaveAttribute('aria-pressed', 'true');
    await expect(
      section.getByRole('button', { name: loc.chipLiq, exact: true }),
    ).toHaveAttribute('aria-pressed', 'false');

    // step 4 — scroll on; the reading has only ever GAINED detail
    await section.getByText(loc.stepLiq).first().scrollIntoViewIfNeeded();
    await expect(section.locator('[aria-current="step"]')).toContainText(loc.stepLiqText);
    for (const chip of [loc.chipStr, loc.chipOb, loc.chipFvg, loc.chipLiq]) {
      await expect(
        section.getByRole('button', { name: chip, exact: true }),
      ).toHaveAttribute('aria-pressed', 'true');
    }
  });

  test('taking a chip ends the choreography for good — scrolling never steals it back', async ({ page }) => {
    const loc = LOCALES[0]!;
    await page.goto(loc.path, { waitUntil: 'domcontentloaded', timeout: 60_000 });
    await dismissCookieBanner(page);

    const section = page.locator('#lecture');
    await section.getByText(loc.stepLiq).first().scrollIntoViewIfNeeded();
    // take control: switch the liquidity layer off
    await section.getByRole('button', { name: loc.chipLiq, exact: true }).click();
    await expect(page.getByText(loc.liqFrag)).toHaveCount(0);

    // scroll back up to the first step: the scroll position must NOT reassert
    // a cumulative state and re-draw the layer the reader just removed.
    await section.getByText(loc.stepBreak).first().scrollIntoViewIfNeeded();
    await expect(page.getByText(loc.liqFrag)).toHaveCount(0);
    await expect(
      section.getByRole('button', { name: loc.chipOb, exact: true }),
    ).toHaveAttribute('aria-pressed', 'true');
  });
});
