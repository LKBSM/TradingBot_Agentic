import { expect, test, type Locator, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * DATA-4 (2026-09-27) — tout ce que cette spec affirmait du catalogue était
 * écrit en dur : « 98 » entrées d'affichage, six groupes peuplés, et BTCUSD pris
 * comme exemple de marché NON couvert. Le registre est passé de 2 à 80 marchés,
 * donc 79 entrées du catalogue sont devenues RÉELLES : il n'en reste que 21
 * (20 indices, écartés faute d'horaires de place, et 1 crypto), quatre groupes
 * sont vides, et BTCUSD est maintenant un marché suivi — le test de « l'état
 * vide sur un marché sans donnée » ne testait donc plus rien du tout.
 *
 * Tout est désormais DÉRIVÉ des deux fichiers de configuration, comme le fait
 * déjà la garde vitest équivalente. Le prochain changement de périmètre n'aura
 * rien à réécrire ici.
 */
const REPO_ROOT = resolve(process.cwd(), '..');

interface CatalogEntry {
  id: string;
  label: string;
  group: string;
}

const CATALOG_ENTRIES: CatalogEntry[] = JSON.parse(
  readFileSync(resolve(REPO_ROOT, 'config/market_catalog_ux_test.json'), 'utf-8'),
).markets;

const REAL_IDS = new Set<string>(
  JSON.parse(readFileSync(resolve(REPO_ROOT, 'config/markets.json'), 'utf-8')).markets.map(
    (m: { id: string }) => m.id,
  ),
);

/** Les entrées d'affichage seulement — celles que le moteur ne suit PAS. */
const CATALOG_ONLY = CATALOG_ENTRIES.filter((e) => !REAL_IDS.has(e.id));
/** Seuls les groupes qui ont encore une entrée d'affichage sont rendus. */
const CATALOG_ONLY_GROUPS = [...new Set(CATALOG_ONLY.map((e) => e.group))];
/** Un marché réellement non couvert, pour l'état vide honnête. */
const UNCOVERED = CATALOG_ONLY[0]!;

/**
 * MKT-1 — market-catalogue UX test, live capture at both viewports.
 *
 * ⚠️ REQUIRES THE FLAG ON. The catalogue is read at build time, so the server
 * under test must have been built AND started with
 * NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST=1. What this captures is therefore a
 * TEST state, never the default one — with the flag off the catalogue does not
 * appear at all (asserted by the vitest suite, not here).
 *
 *   cd webapp
 *   NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST=1 npm run build
 *   PORT=3123 NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST=1 npm start
 *   E2E_BASE_URL=http://localhost:3123 npx playwright test mkt1-catalog \
 *     --project=chromium-desktop --workers=1
 *
 * Both viewports are emulated HERE (test.use), so run the desktop project only —
 * running both projects would execute every case twice for nothing.
 *
 * Desktop uses the always-open rail (`mkt-selector-rail`); the phone uses the
 * MobileWorkspace "Marchés" tab (`mkt-selector-panel`), which is the default tab
 * and switches itself to "Lecture" once a market is picked. Locators are scoped
 * to the right variant rather than using .first(), so a hidden twin in the DOM
 * can never satisfy an assertion.
 */
const DIR = '../docs/audits/mkt-1-shots';

const START = Math.floor(Date.UTC(2026, 0, 1) / 1000);
const CANDLES = Array.from({ length: 300 }, (_, i) => {
  const close = 2000 + i;
  return { time: START + i * 900, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
});

/** Only the REAL market is mocked — a catalogue market must never be requested. */
async function mockReal(page: Page) {
  await page.route('**/api/candles**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ instrument: 'XAUUSD', timeframe: 'M15', candles: CANDLES, has_more_history: false }),
    }),
  );
  await page.route('**/api/market-reading**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FIXTURE_XAU_M15) }),
  );
  await page.route('**/api/market-status**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) }),
  );
}

const VIEWPORTS = [
  { w: 1280, h: 800, tag: 'desktop', variant: 'rail' },
  { w: 390, h: 844, tag: 'mobile', variant: 'panel' },
] as const;

async function openApp(page: Page) {
  await mockReal(page);
  await page.goto('/fr/app?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
  await dismissCookieBanner(page);
  await page.waitForTimeout(1800);
}

for (const vp of VIEWPORTS) {
  test.describe(`mkt-1 ${vp.tag}`, () => {
    test.use({ viewport: { width: vp.w, height: vp.h } });

    /** The market selector for THIS viewport. */
    const selector = (page: Page): Locator => page.getByTestId(`mkt-selector-${vp.variant}`);

    test(`catalogue visible et groupé — ${vp.tag}`, async ({ page }) => {
      test.setTimeout(90_000);
      await openApp(page);

      const col = selector(page);
      await expect(col).toBeVisible();
      // The mode announces itself, and says the real ratio.
      const banner = col.getByTestId('mkt-uxtest-banner');
      await expect(banner).toBeVisible();
      await expect(banner).toContainText('Mode test UX');
      await expect(banner).toContainText(String(CATALOG_ONLY.length));
      // Real markets stay visibly apart from the display-only ones.
      await expect(col.getByText('Suivis par le moteur')).toBeVisible();

      await page.screenshot({ path: `${DIR}/catalogue-${vp.tag}.png`, fullPage: true });
    });

    test(`catégories repliées par défaut, dépliables — ${vp.tag}`, async ({ page }) => {
      test.setTimeout(90_000);
      await openApp(page);

      const col = selector(page);
      // Chaque catégorie qui a ENCORE une entrée d'affichage est rendue, et
      // repliée à l'arrivée. Les groupes vidés par DATA-4 n'apparaissent pas :
      // c'est le comportement voulu, pas une régression.
      for (const g of CATALOG_ONLY_GROUPS) {
        await expect(col.getByTestId(`mkt-group-head-${g}`)).toHaveAttribute('aria-expanded', 'false');
      }

      const head = col.getByTestId(`mkt-group-head-${UNCOVERED.group}`);
      await head.click();
      await expect(head).toHaveAttribute('aria-expanded', 'true');
      await expect(col.getByRole('button', { name: UNCOVERED.label })).toBeVisible();

      await page.waitForTimeout(400);
      await page.screenshot({ path: `${DIR}/categorie-ouverte-${vp.tag}.png`, fullPage: true });
    });

    test(`état vide honnête sur un marché sans donnée — ${vp.tag}`, async ({ page }) => {
      test.setTimeout(90_000);

      // Any DATA request for an uncovered market is a failure of the whole
      // mission. Scoped to /api/ on purpose: navigating to
      // /app?instrument=<id> (and its Next RSC payload) is exactly how the
      // market gets selected — that URL naming the market is the feature, not
      // a leak. What must never happen is a candle/reading/price call.
      const leaked: string[] = [];
      page.on('request', (req) => {
        const url = req.url();
        // Uniquement des ids RÉELLEMENT non couverts : BTCUSD et XAGUSD sont
        // devenus des marchés suivis, les demander est désormais légitime.
        if (url.includes('/api/') && new RegExp(UNCOVERED.id).test(url)) leaked.push(url);
      });

      await openApp(page);
      const col = selector(page);
      await col.getByTestId(`mkt-group-head-${UNCOVERED.group}`).click();
      await col.getByRole('button', { name: UNCOVERED.label }).click();
      // Picking a market writes it into the URL (state → URL, AppWorkspace) and
      // triggers an RSC navigation; on the phone it also switches to the
      // "Lecture" tab. Wait for the URL rather than a fixed delay — a sleep long
      // enough on an idle machine is not long enough on a busy one.
      await page.waitForURL(new RegExp(`instrument=${UNCOVERED.id}`), { timeout: 15_000 });

      await expect(page.getByText('Pas encore disponible sur ce marché')).toBeVisible({
        timeout: 15_000,
      });
      await expect(page.getByText(/rien n’est simulé en attendant/)).toBeVisible();
      // No substitute chart, and no "Réessayer" (retrying cannot change coverage).
      await expect(page.locator('canvas')).toHaveCount(0);
      await expect(page.getByRole('button', { name: /Réessayer/i })).toHaveCount(0);
      expect(leaked, `requests leaked for an uncovered market:\n${leaked.join('\n')}`).toEqual([]);

      await page.screenshot({ path: `${DIR}/etat-vide-${vp.tag}.png`, fullPage: true });
    });

    test(`recherche à 100 entrées — ${vp.tag}`, async ({ page }) => {
      test.setTimeout(90_000);
      await openApp(page);

      const col = selector(page);
      const search = col.getByRole('searchbox');
      const t0 = Date.now();
      await search.fill('indices');
      // A live search opens the groups that have a hit, so nothing stays hidden.
      await expect(col.getByTestId('mkt-group-head-index')).toHaveAttribute('aria-expanded', 'true');
      await expect(col.getByRole('button', { name: 'Nifty 50 (Inde)' })).toBeVisible();
      const elapsed = Date.now() - t0;
      // Generous bound: guards a pathological filter, not a slow machine.
      expect(elapsed, `search took ${elapsed}ms`).toBeLessThan(5000);
      // A category search must not drag in unrelated categories.
      await expect(col.getByTestId('mkt-group-crypto')).toHaveCount(0);

      await page.screenshot({ path: `${DIR}/recherche-${vp.tag}.png`, fullPage: true });
    });
  });
}
