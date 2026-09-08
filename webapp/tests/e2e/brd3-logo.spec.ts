import { expect, test, type Page } from '@playwright/test';
import { dismissCookieBanner } from './utils';

/**
 * BRD-3 — the M.I.A Markets "candle" logo across every surface, in fr, both
 * viewports (the two Playwright projects run each test on desktop 1280×800 and
 * mobile 390×844) and the four themes.
 *
 * TWO COLOURS, on purpose: the candles stay brass (--brand-mark = #C9A14A) on
 * ALL four themes; the wordmark (--brand-word) is white on the three dark themes
 * and near-black (#1A1917) on the light Parchemin/Atelier theme.
 */
const BRASS = 'rgb(201, 161, 74)'; // #C9A14A — candles on every theme
const WORD_DARK = 'rgb(255, 255, 255)'; // #FFFFFF — wordmark on dark themes
const WORD_LIGHT = 'rgb(26, 25, 23)'; // #1A1917 — wordmark on the light theme

const THEMES = [
  { id: 'terminal', word: WORD_DARK },
  { id: 'schema', word: WORD_DARK },
  { id: 'ardoise', word: WORD_DARK },
  { id: 'atelier', word: WORD_LIGHT },
] as const;

async function forceTheme(page: Page, id: string): Promise<void> {
  await page.addInitScript((t) => {
    try {
      window.localStorage.setItem('theme', t as string);
    } catch {
      /* storage unavailable — ignore */
    }
  }, id);
}

function brandHomeLink(page: Page) {
  return page.getByRole('link', { name: /M\.I\.A Markets/i }).first();
}

test.describe('BRD-3 — présence du logo', () => {
  test(`en-tête public : logo cliquable vers l'accueil`, async ({ page }) => {
    await page.goto('/');
    await dismissCookieBanner(page);
    const home = brandHomeLink(page);
    await expect(home).toBeVisible();
    const href = await home.getAttribute('href');
    expect(href).toMatch(/^\/(fr)?\/?$/);
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
    test(`logo présent sur ${path}`, async ({ page }) => {
      await page.goto(path);
      await dismissCookieBanner(page);
      const brand = page
        .getByRole('link', { name: /M\.I\.A Markets/i })
        .or(page.getByRole('img', { name: /M\.I\.A Markets/i }))
        .first();
      await expect(brand).toBeVisible();
    });
  }
});

test.describe('BRD-3 — bougies laiton sur les 4 thèmes, nom blanc/noir selon le fond', () => {
  // The horizontal lockup (candles + wordmark) lives in the public header at
  // desktop width; assert its resolved colours per theme.
  test.use({ viewport: { width: 1280, height: 800 } });

  for (const { id, word } of THEMES) {
    test(`thème ${id} : bougies laiton + nom ${word === WORD_LIGHT ? 'noir' : 'blanc'}`, async ({
      page,
    }) => {
      await forceTheme(page, id);
      await page.goto('/');
      await dismissCookieBanner(page);
      await expect(page.locator('html')).toHaveAttribute('data-design', id);

      const logo = page.locator('header a[aria-label*="M.I.A Markets"] svg').first();
      // Candles: brass on EVERY theme (a leaf rect inherits --brand-mark).
      await expect(logo.locator('rect').first()).toHaveCSS('fill', BRASS);
      // Wordmark: --brand-word, white on dark / near-black on light — never brass.
      const textFill = logo.locator('text').first();
      await expect(textFill).toHaveCSS('fill', word);
      expect(word).not.toBe(BRASS);
    });
  }
});

test.describe('BRD-3 — le logo ne décore jamais une erreur', () => {
  test('la page 404 est textuelle, sans logo dans le contenu', async ({ page }) => {
    await page.goto('/fr/cette-page-nexiste-pas-1234', { waitUntil: 'domcontentloaded' });
    await expect(
      page.getByRole('heading', { name: /introuvable|not be found|404/i }).first(),
    ).toBeVisible();
    await expect(page.getByRole('img', { name: /M\.I\.A Markets/i })).toHaveCount(0);
    await expect(page.getByRole('link', { name: /M\.I\.A Markets/i })).toHaveCount(0);
  });
});

test.describe('BRD-3 — avatar de M.I.A dans la conversation', () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test("l'avatar (bougies compactes) est à côté des messages de l'agent", async ({ page }) => {
    test.setTimeout(90_000);
    await page.goto('/app', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);
    const avatar = page.getByTestId('mia-avatar').first();
    await expect(avatar).toBeVisible({ timeout: 30_000 });
    // The compact mark uses a wick at x1="59", unique to the three-candle variant.
    await expect(avatar.locator('svg line[x1="59"]')).toHaveCount(1);
  });
});
