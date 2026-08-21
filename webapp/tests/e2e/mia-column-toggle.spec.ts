import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * M.I.A disposition toggle (/app) — the user switches M.I.A between two layouts
 * from the page itself (desktop ≥1280, persisted in localStorage):
 *   · COLUMN mode → chat docked as the right column; a header button reduces it.
 *   · BUBBLE mode → chat reduced to the floating bubble (`.chat-fab`), the centre
 *     reclaims the full width; the bubble reopens the SAME drawer, whose header
 *     carries the "dock to column" button.
 * A single coherent state drives both — there is no separate hide/reopen path.
 *
 * Below 1280 the shell keeps its own responsive behaviour (tablet drawer / phone
 * tab), so the desktop toggle must not leak there. The reading endpoints are
 * mocked (the prod build proxies to a backend absent under test). Locale fr-FR.
 */
const COLLAPSE = 'Réduire en bulle'; // column → bubble (chat.collapseToBubble)
const DOCK = 'Afficher en colonne'; // bubble → column (chat.dockToColumn)
const FAB = "Ouvrir l'assistant"; // the bubble (chat.openPanel)
// Playwright runs with cwd = webapp; the audit shots live at the repo-root
// docs/audits (one level up).
const SHOTS = '../docs/audits/mia-bubble-shots';

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

test.describe('M.I.A disposition toggle — desktop 1280×800', () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test.beforeEach(async ({ page }) => {
    await mockReading(page);
  });

  test('column by default → bubble → back, conversation subtree preserved', async ({ page }) => {
    await page.goto('/app?instrument=XAUUSD&timeframe=M15');
    await dismissCookieBanner(page);

    // (a) COLUMN by default: docked header shows, the "reduce to bubble" button is
    // present, and there is no floating bubble yet (the fab is display:none, so
    // the ARIA query — which skips display:none subtrees — finds none).
    await expect(page.getByText('M.I.A Agent', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: COLLAPSE })).toBeVisible();
    await expect(page.getByRole('button', { name: FAB })).toHaveCount(0);

    const centre = page.locator('#main.center');
    const widthColumn = (await centre.boundingBox())!.width;

    // Mark a node INSIDE the chat subtree (set imperatively, not via React). If a
    // toggle unmounted/remounted the chat, this DOM node — and its marker — would
    // be recreated and the attribute lost. Its survival proves the conversation
    // state is never torn down by switching disposition.
    await page.getByText('M.I.A Agent', { exact: true }).evaluate((el) =>
      el.closest('aside')?.setAttribute('data-mia-persist', '1'),
    );
    await page.screenshot({ path: `${SHOTS}/a-column-default.png` });

    // (b) Reduce to bubble → the centre reclaims the ~338px track and the floating
    // bubble appears.
    await page.getByRole('button', { name: COLLAPSE }).click();
    const fab = page.getByRole('button', { name: FAB });
    await expect(fab).toBeVisible();
    const widthBubble = (await centre.boundingBox())!.width;
    expect(widthBubble).toBeGreaterThan(widthColumn + 200);
    await page.screenshot({ path: `${SHOTS}/b-bubble.png` });

    // (c) The bubble reopens the SAME chat as a drawer: M.I.A's header + the honesty
    // note render identically, and the header now offers "dock to column".
    await fab.click();
    await expect(page.getByText('M.I.A Agent', { exact: true })).toBeVisible();
    await expect(page.getByText(/pédagogique/i).first()).toBeVisible();
    const dockBtn = page.getByRole('button', { name: DOCK });
    await expect(dockBtn).toBeVisible();
    await page.screenshot({ path: `${SHOTS}/c-bubble-open.png` });

    // The chat subtree was never remounted across (a)→(c): the marker survived.
    await expect(page.locator('aside[data-mia-persist="1"]')).toHaveCount(1);

    // (d) Dock back to column → the bubble disappears, the column returns, and the
    // centre is back to its original width. Marker still intact.
    await dockBtn.click();
    await expect(page.getByRole('button', { name: FAB })).toHaveCount(0);
    await expect(page.getByRole('button', { name: COLLAPSE })).toBeVisible();
    const widthReopened = (await centre.boundingBox())!.width;
    expect(Math.abs(widthReopened - widthColumn)).toBeLessThanOrEqual(2);
    await expect(page.locator('aside[data-mia-persist="1"]')).toHaveCount(1);
    await page.screenshot({ path: `${SHOTS}/d-column-back.png` });
  });

  test('bubble disposition persists across a reload', async ({ page }) => {
    await page.goto('/app?instrument=XAUUSD&timeframe=M15');
    await dismissCookieBanner(page);
    await page.getByRole('button', { name: COLLAPSE }).click();
    await expect(page.getByRole('button', { name: FAB })).toBeVisible();

    await page.reload();
    // The persisted choice re-applies: the page comes back in bubble mode (the
    // floating bubble is shown; the column is not docked).
    await expect(page.getByRole('button', { name: FAB })).toBeVisible();
    await expect(page.getByRole('button', { name: COLLAPSE })).toHaveCount(0);
  });
});

test.describe('M.I.A disposition toggle — mobile 390×844', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test.beforeEach(async ({ page }) => {
    await mockReading(page);
  });

  test('the desktop disposition toggle does not leak on phone', async ({ page }) => {
    await page.goto('/app?instrument=XAUUSD&timeframe=M15');
    await dismissCookieBanner(page);
    // The phone layout (MobileWorkspace tabs) owns the chat; the shell chat column,
    // its bubble and the disposition toggle are display:none there — so none of
    // them appears in the accessibility tree.
    await expect(page.getByRole('button', { name: COLLAPSE })).toHaveCount(0);
    await expect(page.getByRole('button', { name: DOCK })).toHaveCount(0);
    await expect(page.getByRole('button', { name: FAB })).toHaveCount(0);
    await page.screenshot({ path: `${SHOTS}/e-mobile.png` });
  });
});
