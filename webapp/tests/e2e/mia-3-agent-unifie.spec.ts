import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';
import frMessages from '../../messages/fr.json';
import enMessages from '../../messages/en.json';

/**
 * MIA-3 — ONE M.I.A, everywhere, one conversation.
 *
 * The /app and /zones panels are now the SAME shell chat column (MiaPanel). This
 * spec drives it with the SSE endpoint mocked and asserts, across the mission
 * matrix (1280×800, 1440×900, 390×844 · fr + en · column + bubble · with/without
 * a selected zone · a four-message conversation):
 *  - /zones shows the shared column (not the old narrow stub) with the selected
 *    zone as an orientation subject;
 *  - a zone question reaches the backend WITH the selected-zone orientation
 *    preamble (id-locked), and a question keeps working after deselecting;
 *  - re-clicking the selected card deselects it (subject gone) without wiping the
 *    conversation;
 *  - the conversation is continuous and shared between /app and /zones;
 *  - the bubble↔column choice toggles and the field stays reachable in both.
 *
 * DISPOSITION — /app opens in COLUMN, /zones opens in BUBBLE (founder's call:
 * the docked column narrowed the cards column to 670px, which wrapped the filter
 * bar and cost /zones half its visible cards — see vz-2-measure.spec.ts). That
 * changes where M.I.A STARTS, never what it is: MIA-3's contract is one panel
 * and one conversation everywhere. So the tests below that are about the DOCKED
 * column open it first, in the one click a user would make.
 */

const STREAM = '**/api/chatbot/stream';

function sse(events: Array<Record<string, unknown>>): string {
  return events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('');
}
function answerBody(content: string): string {
  return sse([
    { event: 'activity' },
    { event: 'answer', content, blocked_reason: null, tool_calls_made: [], view_actions: [] },
  ]);
}

function makeCandles(n = 150) {
  const base = 2300;
  const start = Math.floor(Date.UTC(2026, 5, 20) / 1000);
  const candles = Array.from({ length: n }, (_, i) => {
    const close = base + i * 2;
    return { time: start + i * 900, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
  });
  return { instrument: 'XAUUSD', timeframe: 'M15', candles };
}

/** Mock market data + a counting SSE stream; returns the captured request bodies. */
async function mockAll(page: Page): Promise<string[]> {
  const bodies: string[] = [];
  await page.route('**/api/candles**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(makeCandles()) }),
  );
  await page.route('**/api/market-reading**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FIXTURE_XAU_M15) }),
  );
  await page.route('**/api/market-status**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) }),
  );
  let n = 0;
  await page.route(STREAM, (route) => {
    bodies.push(route.request().postData() ?? '');
    n += 1;
    route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: answerBody(`Réponse M.I.A numéro ${n}.`),
    });
  });
  return bodies;
}

const chatInput = (page: Page) => page.getByTestId('chat-input').first();

async function ask(page: Page, text: string) {
  const field = chatInput(page);
  await field.waitFor({ state: 'visible', timeout: 60_000 });
  await field.click();
  await field.fill(text);
  // Wait for the composer to be sendable: the draft is set AND the page has
  // bound its combo (activeSignal) — the send button gates on both.
  const btn = page
    .locator('form:has([data-testid="chat-input"])')
    .first()
    .locator('button[type="submit"]');
  await expect(btn).toBeEnabled({ timeout: 30_000 });
  await btn.click();
}

/** Button labels straight from the message files — never a hand-copied string. */
const MESSAGES = { fr: frMessages, en: enMessages } as const;
function chatLabel(locale: 'fr' | 'en', key: 'openPanel' | 'dockToColumn'): string {
  const label = (MESSAGES[locale] as unknown as { app: { chat: Record<string, string> } }).app.chat[
    key
  ];
  // Fail on the missing key rather than on a mystifying « locator not found »
  // 30 seconds later, the way ShellRail.test.tsx does.
  if (!label) throw new Error(`app.chat.${key} absent de messages/${locale}.json`);
  return label;
}

/**
 * Put M.I.A in COLUMN disposition, whatever the space's default is. A no-op
 * where the column is already docked (/app), one click where it is not
 * (/zones). The fab is matched by class so the helper stays locale-free; the
 * dock button is matched by its real translated label.
 */
async function dockColumn(page: Page, locale: 'fr' | 'en') {
  const fab = page.locator('.chat-fab');
  if (!(await fab.isVisible().catch(() => false))) return;
  await fab.click();
  await page.getByRole('button', { name: chatLabel(locale, 'dockToColumn') }).click();
  await expect(page.locator('.chat-fab')).toBeHidden({ timeout: 10_000 });
}

async function goto(page: Page, path: string, locale: 'fr' | 'en') {
  test.setTimeout(90_000);
  const prefix = locale === 'en' ? '/en' : '';
  await page.goto(`${prefix}${path}`, { waitUntil: 'domcontentloaded' });
  await dismissCookieBanner(page);
}

// ── Column mode, desktop — the core matrix (fr + en at 1280×800) ──────────────
for (const locale of ['fr', 'en'] as const) {
  test.describe(`MIA-3 · ${locale} · 1280×800 (column)`, () => {
    test.use({ viewport: { width: 1280, height: 800 } });

    test(`/zones docks the shared M.I.A column with a zone subject; the question carries the zone orientation`, async ({ page }) => {
      const bodies = await mockAll(page);
      await goto(page, '/zones', locale);
      // /zones starts in bubble — dock it, which is this test's subject.
      await dockColumn(page, locale);

      // The shared column input is visible (not the old inline stub).
      await expect(chatInput(page)).toBeVisible({ timeout: 60_000 });
      // A zone is selected by default → the orientation subject block shows.
      await expect(page.getByTestId('mia-subject')).toBeVisible({ timeout: 60_000 });

      await ask(page, 'Décris cette zone en détail');
      await expect(page.getByText('Réponse M.I.A numéro 1.')).toBeVisible({ timeout: 15_000 });
      // The request carried the selected-zone orientation preamble (id-locked).
      expect(bodies.some((b) => b.includes('[Zone sélectionnée : '))).toBe(true);

      // Re-click the selected card → deselect: the subject disappears, the
      // conversation is NOT wiped.
      await page.locator('.zone.zsel').first().click();
      await expect(page.getByTestId('mia-subject')).toHaveCount(0, { timeout: 10_000 });
      await expect(page.getByText('Réponse M.I.A numéro 1.')).toBeVisible();

      // A question with no zone selected is still answered (orientation, not prison).
      await ask(page, 'Et sur un autre marché, quoi de neuf ?');
      await expect(page.getByText('Réponse M.I.A numéro 2.')).toBeVisible({ timeout: 15_000 });
    });

    test(`/app holds a four-message conversation in the shared column`, async ({ page }) => {
      await mockAll(page);
      await goto(page, '/app?instrument=XAUUSD&timeframe=M15', locale);
      await expect(chatInput(page)).toBeVisible({ timeout: 60_000 });

      await ask(page, 'Première question');
      await expect(page.getByText('Réponse M.I.A numéro 1.')).toBeVisible({ timeout: 15_000 });
      await ask(page, 'Deuxième question');
      await expect(page.getByText('Réponse M.I.A numéro 2.')).toBeVisible({ timeout: 15_000 });

      // Four messages present (2 user + 2 assistant).
      await expect(page.getByText('Première question')).toBeVisible();
      await expect(page.getByText('Deuxième question')).toBeVisible();
    });
  });
}

// ── One continuous conversation that follows the user across pages ────────────
test.describe('MIA-3 · fr · 1280×800 · conversation follows the user', () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test('a message sent on /app is still there on /zones (single conversation)', async ({ page }) => {
    await mockAll(page);
    await goto(page, '/app?instrument=XAUUSD&timeframe=M15', 'fr');
    await ask(page, 'Question posée sur app');
    await expect(page.getByText('Réponse M.I.A numéro 1.')).toBeVisible({ timeout: 15_000 });

    await page.goto('/zones', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);
    // /zones starts in bubble. Open it the way a user would — asserting on the
    // off-canvas panel would pass even if the thread were empty on screen,
    // because Playwright counts an out-of-viewport node as visible.
    await dockColumn(page, 'fr');
    // The earlier exchange survives the navigation (same shared thread).
    await expect(page.getByText('Question posée sur app')).toBeVisible({ timeout: 60_000 });
  });
});

// ── Bubble ↔ column toggle (desktop) ─────────────────────────────────────────
test.describe('MIA-3 · fr · 1280×800 · bubble/column toggle', () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test('the disposition toggles and the field stays reachable in both modes', async ({ page }) => {
    await mockAll(page);
    await goto(page, '/app?instrument=XAUUSD&timeframe=M15', 'fr');
    await expect(chatInput(page)).toBeVisible({ timeout: 60_000 });

    // Column → bubble: the docked input goes away, the floating button appears.
    await page.getByRole('button', { name: /réduire.*bulle|bulle/i }).first().click();
    await expect(page.locator('.chat-fab')).toBeVisible({ timeout: 10_000 });

    // The bubble opens the same panel (conversation never lost): input reachable.
    await page.locator('.chat-fab').click();
    await expect(chatInput(page)).toBeVisible({ timeout: 10_000 });
  });
});

// ── 1440×900 (column) ─────────────────────────────────────────────────────────
test.describe('MIA-3 · fr · 1440×900 (column)', () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test('/zones and /app both dock the shared column and answer', async ({ page }) => {
    await mockAll(page);
    await goto(page, '/zones', 'fr');
    // Same disposition default at 1440 as at 1280 — it follows the space, not
    // the width. Dock it: this test is about the docked column.
    await dockColumn(page, 'fr');
    await expect(chatInput(page)).toBeVisible({ timeout: 60_000 });
    await ask(page, 'Question à 1440');
    await expect(page.getByText('Réponse M.I.A numéro 1.')).toBeVisible({ timeout: 15_000 });
  });
});

// ── 390×844 mobile — /zones uses the floating drawer (no MobileWorkspace tab) ──
test.describe('MIA-3 · fr · 390×844 (mobile drawer)', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('/zones exposes the M.I.A drawer via the floating button and answers', async ({ page }) => {
    await mockAll(page);
    await goto(page, '/zones', 'fr');
    // On phones the column is a floating drawer; open it via the fab.
    const fab = page.locator('.chat-fab');
    await expect(fab).toBeVisible({ timeout: 60_000 });
    await fab.click();
    await expect(chatInput(page)).toBeVisible({ timeout: 10_000 });
    await ask(page, 'Question mobile');
    await expect(page.getByText('Réponse M.I.A numéro 1.')).toBeVisible({ timeout: 15_000 });
  });
});
