import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * APP-1 — the user chooses between BUBBLE and COLUMN for M.I.A, and the choice
 * wins (persisted, never taken back by a resize). The column is offered down to
 * 1100px; below that it would crush the chart, so it is not offered and the
 * control SAYS why. All reading endpoints mocked. Locales fr + en.
 */
const START = Math.floor(Date.UTC(2026, 0, 1) / 1000);
const CANDLES = Array.from({ length: 300 }, (_, i) => {
  const close = 2000 + i;
  return { time: START + i * 900, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
});
const jr = (b: unknown) => ({ status: 200, contentType: 'application/json', body: JSON.stringify(b) });
async function mockReading(page: Page) {
  await page.route('**/api/candles**', (r) => r.fulfill(jr({ instrument: 'XAUUSD', timeframe: 'M15', candles: CANDLES, has_more_history: false })));
  await page.route('**/api/market-reading**', (r) => r.fulfill(jr(FIXTURE_XAU_M15)));
  await page.route('**/api/market-status**', (r) => r.fulfill(jr({})));
}

const L = {
  fr: { collapse: 'Réduire en bulle', dock: 'Afficher en colonne', fab: "Ouvrir l'assistant", needsWidth: 'Colonne disponible sur un écran plus large.', notSynced: "Ton choix d'affichage — non synchronisé." },
  en: { collapse: 'Reduce to a bubble', dock: 'Dock as a column', fab: 'Open the assistant', needsWidth: 'Column available on a wider screen.', notSynced: 'Your display choice — not synced.' },
} as const;

async function goApp(page: Page, locale: 'fr' | 'en') {
  await mockReading(page);
  await page.goto(`/${locale}/app?instrument=XAUUSD&timeframe=M15`);
  await dismissCookieBanner(page).catch(() => {});
}

for (const locale of ['fr', 'en'] as const) {
  const t = L[locale];

  test(`${locale} — laptop band (1152): column ↔ bubble is offered, reachable from both`, async ({ page }) => {
    await page.setViewportSize({ width: 1152, height: 800 });
    await goApp(page, locale);
    // Column by default: the "reduce to bubble" control shows (it used to need ≥1280).
    await expect(page.getByRole('button', { name: t.collapse })).toBeVisible();
    // → bubble
    await page.getByRole('button', { name: t.collapse }).click();
    const fab = page.getByRole('button', { name: t.fab });
    await expect(fab).toBeVisible();
    // Reachable from bubble: open the drawer, the "dock to column" control shows.
    await fab.click();
    await expect(page.getByRole('button', { name: t.dock })).toBeVisible();
    // → back to column
    await page.getByRole('button', { name: t.dock }).click();
    await expect(page.getByRole('button', { name: t.collapse })).toBeVisible();
  });

  test(`${locale} — below 1100 (1000): column not offered, the control explains why`, async ({ page }) => {
    await page.setViewportSize({ width: 1000, height: 800 });
    await goApp(page, locale);
    // No column disposition to toggle to at this width.
    await expect(page.getByRole('button', { name: t.collapse })).toHaveCount(0);
    await expect(page.getByRole('button', { name: t.dock })).toHaveCount(0);
    // Open the bubble; the status explains the column needs a wider screen.
    await page.getByRole('button', { name: t.fab }).click();
    await expect(page.getByText(t.needsWidth)).toBeVisible();
  });

  test(`${locale} — ≥1100 shows the « non synchronisé » mode status`, async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await goApp(page, locale);
    await expect(page.getByText(t.notSynced)).toBeVisible();
  });
}

// ── fr-only behavioural + layout guards ─────────────────────────────────────
test('the chosen mode persists across a reload (bubble at 1152)', async ({ page }) => {
  await page.setViewportSize({ width: 1152, height: 800 });
  await goApp(page, 'fr');
  await page.getByRole('button', { name: L.fr.collapse }).click();
  await expect(page.getByRole('button', { name: L.fr.fab })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('button', { name: L.fr.fab })).toBeVisible();
  await expect(page.getByRole('button', { name: L.fr.collapse })).toHaveCount(0);
});

test('a resize never takes back the user choice', async ({ page }) => {
  await page.setViewportSize({ width: 1400, height: 900 });
  await goApp(page, 'fr');
  // Choose bubble at a wide width.
  await page.getByRole('button', { name: L.fr.collapse }).click();
  await expect(page.getByRole('button', { name: L.fr.fab })).toBeVisible();
  // Resize down (still ≥1100) — the choice stays bubble, NOT reverted to column.
  await page.setViewportSize({ width: 1150, height: 900 });
  await expect(page.getByRole('button', { name: L.fr.fab })).toBeVisible();
  await expect(page.getByRole('button', { name: L.fr.collapse })).toHaveCount(0);
  // Resize back up — still bubble.
  await page.setViewportSize({ width: 1400, height: 900 });
  await expect(page.getByRole('button', { name: L.fr.fab })).toBeVisible();
});

test('conversation subtree AND a typed draft survive the toggle both ways', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await goApp(page, 'fr');
  // Mark the chat subtree imperatively — a remount would drop this attribute.
  await page.getByText('M.I.A Agent', { exact: true }).evaluate((el) =>
    el.closest('aside')?.setAttribute('data-mia-persist', '1'),
  );
  // Type a draft (do NOT send).
  const input = page.locator('aside textarea').first();
  await input.fill('brouillon en cours');
  // column → bubble → open drawer → back to column
  await page.getByRole('button', { name: L.fr.collapse }).click();
  await page.getByRole('button', { name: L.fr.fab }).click();
  await page.getByRole('button', { name: L.fr.dock }).click();
  // The subtree was never torn down, and the draft is intact.
  await expect(page.locator('aside[data-mia-persist="1"]')).toHaveCount(1);
  await expect(page.locator('aside textarea').first()).toHaveValue('brouillon en cours');
});

for (const w of [1280, 1440, 1920]) {
  test(`no horizontal scroll at ${w} (column and bubble)`, async ({ page }) => {
    await page.setViewportSize({ width: w, height: 900 });
    await goApp(page, 'fr');
    const overflowCol = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflowCol, `column overflow @${w}`).toBeLessThanOrEqual(1);
    await page.getByRole('button', { name: L.fr.collapse }).click();
    await page.waitForTimeout(400);
    const overflowBub = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflowBub, `bubble overflow @${w}`).toBeLessThanOrEqual(1);
  });
}

test('the top of « Lecture narrée » is visible without scrolling at 1280×800', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await goApp(page, 'fr');
  const narr = page.locator('.narr').first();
  await expect(narr).toBeVisible();
  const top = (await narr.boundingBox())!.y;
  expect(top).toBeGreaterThan(0);
  expect(top, 'narr top within the first viewport height').toBeLessThan(800);
});

test('the zone count carries its denominator (« N sur M »)', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await goApp(page, 'fr');
  await expect(page.getByText(/\d+\s+sur\s+\d+/).first()).toBeVisible();
});
