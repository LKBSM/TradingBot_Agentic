import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * Chatbot ↔ backend FastAPI e2e (mocked). Drives the LIVE chat that ships in the
 * product — /app's docked AppChatSidebar — with POST /api/chatbot/message mocked
 * per scenario, so the whole frontend pile (ChatInput → ChatProvider →
 * api-client → render) is exercised without a running backend.
 *
 * DETTE-1 repoint: this previously drove the chat via the landing multi-market
 * gallery CTA, which LP-1's home-page refonte removed. It now runs on /app; the
 * chat input/send aria labels (chat.inputAria / chat.sendAria) are shared, so the
 * flow is identical. The /app reading endpoints are mocked too so the page loads
 * without a backend.
 *
 * (npm test = Vitest and does NOT run this dir; the Vitest counterpart is
 *  components/chat/__tests__/chatbot-backend-integration.smoke.test.tsx.)
 */

const CHAT_ENDPOINT = '**/api/chatbot/message';

function makeCandles(n = 150) {
  const base = 2300;
  const start = Math.floor(Date.UTC(2026, 5, 20) / 1000);
  const candles = Array.from({ length: n }, (_, i) => {
    const close = base + i * 2;
    return { time: start + i * 900, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
  });
  return { instrument: 'XAUUSD', timeframe: 'M15', candles };
}

/** Load /app (docked chat) with the reading mocked, ready for the chat input. */
async function gotoAppChat(page: Page) {
  test.setTimeout(90_000);
  await page.route('**/api/candles**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(makeCandles()) }),
  );
  await page.route('**/api/market-reading**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FIXTURE_XAU_M15) }),
  );
  await page.goto('/app?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
  await dismissCookieBanner(page);
}

async function askFreeText(page: Page, text: string) {
  const input = page.getByRole('textbox', { name: /Question libre pour M\.I\.A Agent/i });
  await input.waitFor({ state: 'visible', timeout: 60_000 });
  await input.fill(text);
  await page.getByRole('button', { name: /Envoyer la question/i }).click();
}

// ≥1280px so the chat sidebar is docked (visible without a tab).
test.use({ viewport: { width: 1280, height: 800 } });

test.describe('Chatbot ↔ backend (mocked) — on /app', () => {
  test('Test 1 — réponse non bloquée affichée', async ({ page }) => {
    await page.route(CHAT_ENDPOINT, (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          content: 'XAUUSD H1 est en consolidation sous résistance, ATR contenu.',
          blocked_reason: null,
          tool_calls_made: [],
        }),
      }),
    );

    await gotoAppChat(page);
    await askFreeText(page, 'Décris-moi les conditions XAUUSD H1');

    await expect(page.getByText(/en consolidation sous résistance/i)).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText('Question recadrée')).toHaveCount(0);
  });

  test('Test 2 — demande d’action → refus + indicateur blocked_reason', async ({ page }) => {
    await page.route(CHAT_ENDPOINT, (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          content: 'Je décris les conditions du marché. La décision d’agir t’appartient.',
          blocked_reason: 'trade_request',
          tool_calls_made: [],
        }),
      }),
    );

    await gotoAppChat(page);
    await askFreeText(page, 'Dois-je acheter EURUSD ?');

    await expect(page.getByText(/La décision d’agir/i)).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText('Question recadrée')).toBeVisible();
  });

  test('Test 3 — backend 503 → message fallback user-friendly', async ({ page }) => {
    await page.route(CHAT_ENDPOINT, (route) =>
      route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Chatbot service not configured' }),
      }),
    );

    await gotoAppChat(page);
    await askFreeText(page, 'Bonjour, en bref ?');

    await expect(page.getByText(/mode chatbot en direct n'est pas disponible/i)).toBeVisible({ timeout: 10_000 });
  });
});
