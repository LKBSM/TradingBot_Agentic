import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * M.I.A column toggle (/app) — the docked chat column is visible by default on
 * desktop (≥1280) and can be collapsed to give the chart more room, then
 * reopened. The choice persists across reloads (localStorage). Below 1280 the
 * shell keeps its own responsive behaviour (tablet drawer / phone tab), so the
 * desktop hide/reopen affordances must not leak there.
 *
 * The reading endpoints are mocked (the prod build proxies to a backend that is
 * absent under test), so the reading — and its apphead reopen affordance —
 * render deterministically. Locale is fr-FR (playwright.config).
 */
const HIDE = 'Masquer le panneau M.I.A';
const SHOW = 'Afficher le panneau M.I.A';
// Playwright runs with cwd = webapp; the audit shots live at the repo-root
// docs/audits (one level up).
const SHOTS = '../docs/audits/mia-column-shots';

const START = Math.floor(Date.UTC(2026, 0, 1) / 1000);
const CANDLES = Array.from({ length: 300 }, (_, i) => {
  const close = 2000 + i;
  return { time: START + i * 900, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
});

async function mockReading(page: Page) {
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

test.describe('M.I.A column toggle — desktop 1280×800', () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test.beforeEach(async ({ page }) => {
    await mockReading(page);
  });

  test('visible by default, collapses, reopens', async ({ page }) => {
    await page.goto('/app?instrument=XAUUSD&timeframe=M15');
    await dismissCookieBanner(page);

    // (a) Column docked and visible by default at load — the M.I.A header shows,
    // the collapse button is present, and there is no reopen affordance yet.
    await expect(page.getByText('M.I.A Agent', { exact: true })).toBeVisible();
    const hideBtn = page.getByRole('button', { name: HIDE });
    await expect(hideBtn).toBeVisible();
    await expect(page.getByRole('button', { name: SHOW })).toHaveCount(0);

    const centre = page.locator('#main.center');
    const widthOpen = (await centre.boundingBox())!.width;
    await page.screenshot({ path: `${SHOTS}/a-column-default.png` });

    // (b) Collapse → the column disappears, the centre reclaims the width, and a
    // reopen affordance appears in the apphead.
    await hideBtn.click();
    await expect(page.getByText('M.I.A Agent', { exact: true })).toBeHidden();
    const showBtn = page.getByRole('button', { name: SHOW });
    await expect(showBtn).toBeVisible();
    const widthCollapsed = (await centre.boundingBox())!.width;
    expect(widthCollapsed).toBeGreaterThan(widthOpen + 200); // ~338px column reclaimed
    await page.screenshot({ path: `${SHOTS}/b-column-hidden.png` });

    // (c) Reopen from the collapsed state → column back, reopen button gone,
    // centre back to its original width.
    await showBtn.click();
    await expect(page.getByText('M.I.A Agent', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: SHOW })).toHaveCount(0);
    const widthReopened = (await centre.boundingBox())!.width;
    expect(Math.abs(widthReopened - widthOpen)).toBeLessThanOrEqual(2);
    await page.screenshot({ path: `${SHOTS}/c-column-reopened.png` });
  });

  test('collapsed state persists across a reload', async ({ page }) => {
    await page.goto('/app?instrument=XAUUSD&timeframe=M15');
    await dismissCookieBanner(page);
    await page.getByRole('button', { name: HIDE }).click();
    await expect(page.getByText('M.I.A Agent', { exact: true })).toBeHidden();

    await page.reload();
    // The persisted choice re-applies: column stays hidden, reopen affordance present.
    await expect(page.getByRole('button', { name: SHOW })).toBeVisible();
    await expect(page.getByText('M.I.A Agent', { exact: true })).toBeHidden();
  });
});

test.describe('M.I.A column toggle — mobile 390×844', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test.beforeEach(async ({ page }) => {
    await mockReading(page);
  });

  test('desktop hide/reopen affordances do not leak on phone', async ({ page }) => {
    await page.goto('/app?instrument=XAUUSD&timeframe=M15');
    await dismissCookieBanner(page);
    // The phone layout (MobileWorkspace tabs) owns the chat; the desktop docked
    // column and its toggle are not part of it.
    await expect(page.getByRole('button', { name: HIDE })).toHaveCount(0);
    await expect(page.getByRole('button', { name: SHOW })).toHaveCount(0);
    await page.screenshot({ path: `${SHOTS}/d-mobile.png` });
  });
});
