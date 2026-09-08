import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * MIA-2 — the three latency states of one M.I.A turn, at both viewports.
 *
 * Per the MIA-2 decision the validated prose is never streamed token-by-token
 * (Couche 3 needs the complete text), so perceived speed rests on an honest,
 * immediate activity signal. This spec pins the three visible states of a turn
 * with the answer held back by the mock, at 1280×800 and 390×844:
 *   1. état d'attente        — the thinking indicator is up, no answer yet;
 *   2. réponse en cours      — the answer has just rendered in the thread;
 *   3. réponse terminée      — the indicator is cleared and the input is ready.
 * The honest tool-narration caption ("Reading {market}…") is unit-covered by
 * ThinkingIndicator.test.tsx; here we assert the lifecycle end-to-end.
 */

const STREAM = '**/api/chatbot/stream';
const ANSWER = 'La structure reste haussière, le prix consolide sous une zone.';

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

/** Answer held back ~700 ms so the waiting state is observable before it lands. */
async function mockDelayedAnswer(page: Page, delayMs = 700) {
  await page.route(STREAM, async (route) => {
    await new Promise((r) => setTimeout(r, delayMs));
    route.fulfill({ status: 200, contentType: 'text/event-stream', body: answerBody(ANSWER) });
  });
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

// ── Desktop 1280×800 ────────────────────────────────────────────────────────
test.describe('MIA-2 desktop 1280×800 — three states', () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test('fr — attente → réponse affichée → terminée', async ({ page }) => {
    await mockDelayedAnswer(page);
    await gotoApp(page, 'fr');

    const field = input(page);
    await field.waitFor({ state: 'visible', timeout: 60_000 });
    await field.fill('Décris la structure actuelle');
    // Wait for hydration to enable the send button before clicking (the locator
    // re-queries on each retry, so a re-mount during hydration is handled).
    const submit = form(page).locator('button[type="submit"]');
    await expect(submit).toBeEnabled({ timeout: 10_000 });
    await submit.click();

    // 1) État d'attente: the user message is echoed and the thinking indicator is
    //    up, well before the 700 ms answer — no answer text yet.
    await expect(page.getByText('Décris la structure actuelle')).toBeVisible();
    await expect(page.getByTestId('chat-thinking')).toBeVisible({ timeout: 500 });
    await expect(page.getByText(ANSWER)).toHaveCount(0);
    await expect(input(page)).toBeVisible();

    // 2) Réponse en cours d'affichage: the answer lands and renders in the thread.
    await expect(page.getByText(ANSWER)).toBeVisible({ timeout: 10_000 });

    // 3) Réponse terminée: indicator cleared, input ready for the next turn.
    await expect(page.getByTestId('chat-thinking')).toHaveCount(0);
    await expect(input(page)).toBeVisible();
    await expect(input(page)).toBeEnabled();
  });
});

// ── Mobile 390×844 ──────────────────────────────────────────────────────────
test.describe('MIA-2 mobile 390×844 — three states', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('fr — attente → réponse affichée → terminée (onglet chat)', async ({ page }) => {
    await mockDelayedAnswer(page);
    await gotoApp(page, 'fr');

    // Chat tab (3rd: Marchés · Lecture · Chat) — locale-independent.
    await page.getByRole('tab').nth(2).click();
    const panel = page.getByRole('tabpanel');
    await expect(panel.getByTestId('chat-input')).toBeInViewport();

    await panel.getByTestId('chat-input').fill('Décris la structure');
    await panel.locator('form:has([data-testid="chat-input"]) button[type="submit"]').click();

    // 1) Attente.
    await expect(panel.getByText('Décris la structure')).toBeVisible();
    await expect(panel.getByTestId('chat-thinking')).toBeVisible({ timeout: 500 });
    await expect(panel.getByText(ANSWER)).toHaveCount(0);

    // 2) Réponse affichée.
    await expect(panel.getByText(ANSWER)).toBeVisible({ timeout: 10_000 });

    // 3) Terminée: indicator gone, input still reachable under the answer.
    await expect(panel.getByTestId('chat-thinking')).toHaveCount(0);
    await expect(panel.getByTestId('chat-input')).toBeInViewport();
  });
});
