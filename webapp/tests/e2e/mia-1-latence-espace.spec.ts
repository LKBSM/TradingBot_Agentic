import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * MIA-1 — perceived latency & the space given to the conversation.
 *
 * Drives /app's docked M.I.A Agent with the SSE endpoint mocked, and asserts:
 *  - an activity signal shows almost immediately after send (well before a
 *    delayed answer) — the user is never left staring at a frozen screen;
 *  - at least FOUR messages are visible without scrolling at 1280×800;
 *  - the starter suggestions are compact and vanish after the first exchange;
 *  - the input field stays visible in every state.
 * Covered in fr + en at 1280×800, and fr at 390×844.
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

async function mockReading(page: Page) {
  await page.route('**/api/candles**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(makeCandles()) }),
  );
  await page.route('**/api/market-reading**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FIXTURE_XAU_M15) }),
  );
  await page.route('**/api/market-status**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) }),
  );
}

async function gotoApp(page: Page, locale: 'fr' | 'en' = 'fr') {
  test.setTimeout(90_000);
  await mockReading(page);
  const prefix = locale === 'en' ? '/en' : '';
  await page.goto(`${prefix}/app?instrument=XAUUSD&timeframe=M15`, { waitUntil: 'domcontentloaded' });
  await dismissCookieBanner(page);
}

const input = (page: Page) => page.getByTestId('chat-input').first();
const form = (page: Page) => page.locator('form:has([data-testid="chat-input"])').first();

async function ask(page: Page, text: string) {
  const field = input(page);
  await field.waitFor({ state: 'visible', timeout: 60_000 });
  await field.fill(text);
  await form(page).locator('button[type="submit"]').click();
}

// ── Desktop 1280×800 ────────────────────────────────────────────────────────
test.describe('MIA-1 desktop 1280×800', () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  for (const locale of ['fr', 'en'] as const) {
    test(`${locale} — four messages visible without scrolling; suggestions vanish; input always visible`, async ({ page }) => {
      let n = 0;
      await page.route(STREAM, (route) => {
        n += 1;
        const content =
          n === 1
            ? 'La structure reste haussière, le prix consolide sous une zone.'
            : 'Le régime est en expansion, volatilité contenue.';
        route.fulfill({ status: 200, contentType: 'text/event-stream', body: answerBody(content) });
      });

      await gotoApp(page, locale);

      // Empty state: compact starter chips are present, input is visible.
      await expect(input(page)).toBeVisible();
      const starters = page.getByTestId('chat-starter');
      await expect(starters.first()).toBeVisible();
      expect(await starters.count()).toBeGreaterThanOrEqual(2);

      // First exchange.
      await ask(page, 'Décris la structure actuelle');
      await expect(page.getByText('La structure reste haussière, le prix consolide sous une zone.')).toBeVisible({ timeout: 10_000 });
      // Suggestions gone after the first exchange.
      await expect(page.getByTestId('chat-starter')).toHaveCount(0);
      // Input still visible.
      await expect(input(page)).toBeVisible();

      // Second exchange → four turns total.
      await ask(page, 'Et le régime de volatilité ?');
      await expect(page.getByText('Le régime est en expansion, volatilité contenue.')).toBeVisible({ timeout: 10_000 });

      // All FOUR messages are in the viewport (no scrolling required).
      await expect(page.getByText('Décris la structure actuelle')).toBeInViewport();
      await expect(page.getByText('La structure reste haussière, le prix consolide sous une zone.')).toBeInViewport();
      await expect(page.getByText('Et le régime de volatilité ?')).toBeInViewport();
      await expect(page.getByText('Le régime est en expansion, volatilité contenue.')).toBeInViewport();
      // Input remains visible with a full conversation.
      await expect(input(page)).toBeInViewport();
    });
  }

  test('fr — activity signal appears well before a delayed answer', async ({ page }) => {
    await page.route(STREAM, async (route) => {
      // Hold the answer back so the activity signal must carry the wait.
      await new Promise((r) => setTimeout(r, 900));
      route.fulfill({ status: 200, contentType: 'text/event-stream', body: answerBody('Réponse après délai.') });
    });
    await gotoApp(page, 'fr');

    await ask(page, 'Décris la structure actuelle');
    // The thinking indicator shows almost immediately — long before the 900 ms answer.
    await expect(page.getByTestId('chat-thinking')).toBeVisible({ timeout: 500 });
    await expect(input(page)).toBeVisible();
    // Then the answer lands and the indicator clears.
    await expect(page.getByText('Réponse après délai.')).toBeVisible({ timeout: 10_000 });
    await expect(page.getByTestId('chat-thinking')).toHaveCount(0);
  });
});

// ── Mobile 390×844 ──────────────────────────────────────────────────────────
test.describe('MIA-1 mobile 390×844', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('fr — chat tab: input visible, suggestions compact, a message renders', async ({ page }) => {
    await page.route(STREAM, (route) =>
      route.fulfill({ status: 200, contentType: 'text/event-stream', body: answerBody('Lecture mobile de la structure.') }),
    );
    await gotoApp(page, 'fr');

    // Select the Chat tab (3rd tab: Marchés · Lecture · Chat) — locale-independent.
    await page.getByRole('tab').nth(2).click();

    // Scope to the visible mobile chat tabpanel — at <768 the shell ALSO mounts a
    // hidden copy of the sidebar sharing the same turns, so global text selectors
    // would match twice.
    const panel = page.getByRole('tabpanel');
    await expect(panel.getByTestId('chat-input')).toBeInViewport();
    await expect(panel.getByTestId('chat-starter').first()).toBeVisible();

    await panel.getByTestId('chat-input').fill('Décris la structure');
    await panel.locator('form:has([data-testid="chat-input"]) button[type="submit"]').click();
    await expect(panel.getByText('Lecture mobile de la structure.')).toBeVisible({ timeout: 10_000 });
    // Input stays reachable under the answer.
    await expect(panel.getByTestId('chat-input')).toBeInViewport();
  });
});
