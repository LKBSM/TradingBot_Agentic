import { expect, test } from '@playwright/test';

/**
 * Chantier 5.A — e2e de l'intégration chatbot webapp ↔ backend FastAPI.
 *
 * Le backend FastAPI n'est PAS requis : on intercepte POST /api/chatbot/message
 * avec page.route() et on renvoie des réponses simulées. Cela teste toute la
 * pile frontend (ChatInput → ChatProvider → api-client → rendu) sans dépendre
 * d'un service externe, donc reproductible en CI.
 *
 * Prérequis (une fois) :  npx playwright install chromium
 * Lancer :                npm run test:e2e
 *
 * (npm test = Vitest et N'EXÉCUTE PAS ce dossier ; le pendant Vitest vérifié
 *  est components/chat/__tests__/chatbot-backend-integration.smoke.test.tsx.)
 */

const CHAT_ENDPOINT = '**/api/chatbot/message';

async function openChatPanel(page: import('@playwright/test').Page) {
  await page.goto('/#multi-marche');
  await page
    .getByRole('button', {
      name: /Ouvrir le chatbot pour poser une question contextuelle/i,
    })
    .first()
    .click();
  await expect(page.getByRole('dialog')).toBeVisible();
}

async function askFreeText(page: import('@playwright/test').Page, text: string) {
  const input = page.getByRole('textbox', { name: /Question libre pour Sentinel/i });
  await expect(input).toBeVisible();
  await input.fill(text);
  await page.getByRole('button', { name: /Envoyer la question/i }).click();
}

test.describe('Chatbot ↔ backend Chantier 4 (mocked)', () => {
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

    await openChatPanel(page);
    await askFreeText(page, 'Décris-moi les conditions XAUUSD H1');

    await expect(
      page.getByText(/en consolidation sous résistance/i),
    ).toBeVisible({ timeout: 10_000 });
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

    await openChatPanel(page);
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

    await openChatPanel(page);
    await askFreeText(page, 'Bonjour, en bref ?');

    await expect(
      page.getByText(/mode chatbot en direct n'est pas disponible/i),
    ).toBeVisible({ timeout: 10_000 });
  });
});
