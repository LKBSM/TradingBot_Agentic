import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import frMessages from '@/messages/fr.json';
import { dismissCookieBanner } from './utils';

/**
 * MKT-1 — the reusable MarketSelector, adopted on /app (rail + mobile panel) and
 * /zones (header bar), fed by the single market registry (config/markets.json).
 *
 * Assertions are behaviour + presence (language-anchored on the FR names the
 * MESSAGES ship — `calendar.market.*`, not the registry's baseline labels, since
 * #240 made market names follow the page language), covering both required
 * viewports (1280×800, 390×844). Each
 * MarketSelector instance is scoped via its data-testid so a market label that
 * also renders elsewhere (reading header, freshbox) never confuses the check:
 *   · the registry markets (Or, Euro) render — no phantom market;
 *   · search filters to the matching market; a no-match query shows the explicit
 *     "aucun marché ne correspond" message (never a silent fallback);
 *   · a market can be pinned (favourite) → surfaces under "Épinglés" + persists;
 *   · on /zones the header bar selector switches the active market.
 * Live data is validated with the founder before merge (repo e2e convention).
 */

function candles() {
  const start = Math.floor(Date.UTC(2026, 5, 20) / 1000);
  return {
    instrument: 'XAUUSD',
    timeframe: 'M15',
    candles: Array.from({ length: 150 }, (_, i) => {
      const close = 2300 + i * 2;
      return { time: start + i * 900, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
    }),
  };
}

async function mockApis(page: Page) {
  await page.route('**/api/access/me', (r) =>
    r.fulfill({ json: { has_full_access: true, entitlements: { instruments: [], timeframes: [] } } }),
  );
  await page.route('**/api/candles**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(candles()) }),
  );
  await page.route('**/api/market-reading**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FIXTURE_XAU_M15) }),
  );
  await page.route('**/api/market-status**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) }),
  );
  await page.route('**/api/latest-price**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ price: 2387.4, ts: Date.now() / 1000 }) }),
  );
}

// Les noms de marchés viennent de `calendar.market.*` depuis #240 (« les noms de
// marchés suivent la langue de la page ») : une paire de change s'affiche sous son
// code neutre (« EUR/USD »), un métal sous son nom traduit (« Or » / « Gold »).
// Ces constantes codaient « Euro / Dollar (EUR/USD) » en dur et sont devenues
// fausses à cette fusion. Elles lisent maintenant la MÊME source que le composant :
// ce qui est testé ici, c'est que le bon marché est listé, sélectionné, épinglé —
// pas l'orthographe du jour de son nom, qui appartient à la traduction.
const OR = frMessages.calendar.market.XAUUSD;
const EURO = frMessages.calendar.market.EURUSD;

// ─────────────────────────────────────────────────────────────────────────────
// Desktop 1280×800
// ─────────────────────────────────────────────────────────────────────────────
test.describe('MarketSelector — desktop 1280×800', () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test('/app rail: registry markets + search + pin favourite', async ({ page }) => {
    await mockApis(page);
    await page.goto('/app?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);

    const sel = page.getByTestId('mkt-selector-rail');
    await expect(sel.getByText(OR, { exact: true })).toBeVisible();
    await expect(sel.getByText(EURO, { exact: true })).toBeVisible();

    // Search filters to the matching market only.
    const search = sel.getByLabel(/Rechercher un marché/i);
    await search.fill('euro');
    await expect(sel.getByText(EURO, { exact: true })).toBeVisible();
    await expect(sel.getByText(OR, { exact: true })).toHaveCount(0);

    // No-match query → explicit message, never a silent fallback.
    await search.fill('zzzz-nothing');
    await expect(sel.getByText(/Aucun marché ne correspond/i)).toBeVisible();
    await search.fill('');

    // Pin Gold → it surfaces under "Épinglés" and persists to localStorage.
    //
    // Le nom est EXACT, pas un préfixe. `/Épingler Or/i` visait un registre de
    // deux marchés ; depuis #229 il y en a 80, dont QUATRE commencent par « Or »
    // (Or (XAU/USD), Or en euro, Or en livre, Or en yen), et le motif en
    // attrapait 4 — violation du mode strict. L'étiquette vient de
    // `app.sidebar.pinAria` = « Épingler {combo} ».
    await sel
      .getByRole('button', { name: `Épingler ${OR}`, exact: true })
      .click();
    await expect(sel.getByText('Épinglés')).toBeVisible();
    await expect(sel.getByText('Non synchronisé')).toBeVisible();
    const stored = await page.evaluate(() => window.localStorage.getItem('mia.pinnedMarkets.v1'));
    expect(stored).toContain('XAUUSD');

    await page.screenshot({ path: '../docs/audits/mkt1-shots/app-rail-desktop.png', fullPage: false });
  });

  test('/zones header bar: switch the active market', async ({ page }) => {
    await mockApis(page);
    await page.goto('/zones?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);

    const bar = page.getByTestId('mkt-selector-bar');
    const trigger = bar.getByRole('button', { name: 'Marchés', exact: true });
    await expect(trigger).toContainText(OR);
    await trigger.click();

    await bar.getByRole('list').getByText(EURO, { exact: true }).click();

    // The bar now reflects the selected market and the URL followed. Le
    // déclencheur affiche le glyphe COLLÉ au nom (« €EUR/USD ») : on cherche donc
    // le nom tel que les messages le donnent, pas le mot « Euro » qui n'y est plus.
    await expect(trigger).toContainText(EURO);
    await expect(page).toHaveURL(/instrument=EURUSD/);

    await page.screenshot({ path: '../docs/audits/mkt1-shots/zones-bar-desktop.png', fullPage: false });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Mobile 390×844
// ─────────────────────────────────────────────────────────────────────────────
test.describe('MarketSelector — mobile 390×844', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('/app Marchés panel: registry markets + search', async ({ page }) => {
    await mockApis(page);
    await page.goto('/app?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);

    // The Marchés tab is the default on mobile; the panel selector is scoped by testid.
    const sel = page.getByTestId('mkt-selector-panel');
    await expect(sel.getByText(OR, { exact: true })).toBeVisible();
    await expect(sel.getByText(EURO, { exact: true })).toBeVisible();

    const search = sel.getByLabel(/Rechercher un marché/i);
    await search.fill('or');
    await expect(sel.getByText(OR, { exact: true })).toBeVisible();
    await expect(sel.getByText(EURO, { exact: true })).toHaveCount(0);

    await page.screenshot({ path: '../docs/audits/mkt1-shots/app-panel-mobile.png', fullPage: false });
  });

  test('/zones header bar renders on mobile', async ({ page }) => {
    await mockApis(page);
    await page.goto('/zones?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);

    const trigger = page.getByTestId('mkt-selector-bar').getByRole('button', { name: 'Marchés', exact: true });
    await expect(trigger).toBeVisible();
    await expect(trigger).toContainText(OR);

    await page.screenshot({ path: '../docs/audits/mkt1-shots/zones-bar-mobile.png', fullPage: false });
  });
});
