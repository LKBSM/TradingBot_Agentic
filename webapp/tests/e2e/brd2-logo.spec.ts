import { expect, test, type Page } from '@playwright/test';
import { dismissCookieBanner } from './utils';

/**
 * BRD-2 — the M.I.A Markets prism logo across every surface, in both locales,
 * both viewports (the two Playwright projects run each test on desktop 1280×800
 * and mobile iPhone-12 390×844) and both themes.
 *
 * The prism fill is theme-driven via the `--brand-mark` CSS variable, so the
 * same test asserts the correct light/dark variant by reading the resolved
 * colour: #2962FF (rgb 41,98,255) on the light Atelier theme, #7DA3FF
 * (rgb 125,163,255) on the dark themes.
 */

const LIGHT = 'rgb(41, 98, 255)'; // #2962FF
const DARK = 'rgb(125, 163, 255)'; // #7DA3FF

async function forceTheme(page: Page, id: string): Promise<void> {
  // next-themes persists the chosen design under localStorage["theme"] and its
  // pre-paint script applies data-design before first render — no flash.
  await page.addInitScript((t) => {
    try {
      window.localStorage.setItem('theme', t as string);
    } catch {
      /* storage unavailable — ignore */
    }
  }, id);
}

// Home link (Nav on the site chrome, AppHeader on the product shell) carries an
// aria-label containing the company name; every page has exactly one.
function brandHomeLink(page: Page) {
  return page.getByRole('link', { name: /M\.I\.A Markets/i }).first();
}

test.describe('BRD-2 — présence du logo', () => {
  for (const { tag, prefix } of [
    { tag: 'fr', prefix: '' },
    { tag: 'en', prefix: '/en' },
  ]) {
    test(`en-tête public : logo cliquable vers l'accueil (${tag})`, async ({ page }) => {
      await page.goto(`${prefix}/`);
      await dismissCookieBanner(page);
      const home = brandHomeLink(page);
      await expect(home).toBeVisible();
      const href = await home.getAttribute('href');
      expect(href).toMatch(prefix ? /\/en\/?$/ : /^\/(fr)?\/?$/);
    });

    for (const path of [
      '/app',
      '/scanner/decrire',
      '/zones',
      '/actualites',
      '/connexion',
      '/inscription',
      '/abonnement',
    ]) {
      test(`logo présent sur ${path} (${tag})`, async ({ page }) => {
        await page.goto(`${prefix}${path}`);
        await dismissCookieBanner(page);
        // Either the header home link or an in-content lockup (auth/plan pages)
        // — both expose the accessible name "M.I.A Markets".
        const brand = page
          .getByRole('link', { name: /M\.I\.A Markets/i })
          .or(page.getByRole('img', { name: /M\.I\.A Markets/i }))
          .first();
        await expect(brand).toBeVisible();
      });
    }
  }
});

test.describe('BRD-2 — variante claire/sombre automatique', () => {
  for (const { id, color } of [
    { id: 'atelier', color: LIGHT },
    { id: 'terminal', color: DARK },
  ]) {
    test(`le prisme suit le thème ${id}`, async ({ page }) => {
      await forceTheme(page, id);
      await page.goto('/');
      await dismissCookieBanner(page);
      await expect(page.locator('html')).toHaveAttribute('data-design', id);
      const path = page.locator('header a[aria-label*="M.I.A Markets"] svg path').first();
      await expect(path).toHaveCSS('fill', color);
    });
  }
});

test.describe('BRD-2 — le logo ne décore jamais une erreur', () => {
  test('la page 404 est textuelle, sans logo dans le contenu', async ({ page }) => {
    // An unknown URL renders a 404 page (Next's bare page for unmatched routes,
    // or the app's localized not-found for an invalid locale) — either way it is
    // text only and must carry no brand mark.
    await page.goto('/en/cette-page-nexiste-pas-1234', { waitUntil: 'domcontentloaded' });
    await expect(
      page.getByRole('heading', { name: /introuvable|not be found|404/i }).first(),
    ).toBeVisible();
    // No brand lockup used as an error decoration (neither link nor image).
    await expect(page.getByRole('img', { name: /M\.I\.A Markets/i })).toHaveCount(0);
    await expect(page.getByRole('link', { name: /M\.I\.A Markets/i })).toHaveCount(0);
  });
});

test.describe('BRD-2 — avatar de M.I.A dans la conversation', () => {
  // The docked chat sidebar is visible at ≥1280px.
  test.use({ viewport: { width: 1280, height: 800 } });

  test("l'avatar prisme est à côté des messages de l'agent", async ({ page }) => {
    test.setTimeout(90_000);
    await page.goto('/app', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);
    // The agent avatar disc wraps the COMPACT prism — its triangle path is unique
    // to the compact variant, so this locator matches the avatar and nothing else.
    const avatar = page.locator('svg path[d="M48,16 L78,84 L18,84 Z"]').first();
    await expect(avatar).toBeVisible({ timeout: 30_000 });
  });
});
