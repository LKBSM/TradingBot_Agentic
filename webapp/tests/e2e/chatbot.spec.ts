import { expect, test } from '@playwright/test';

test.describe('Chatbot — golden paths', () => {
  test('opens the chat panel when CTA is clicked', async ({ page }) => {
    await page.goto('/#multi-marche');
    // Click "Demander à Sentinel" on the first card.
    await page
      .getByRole('button', {
        name: /Ouvrir le chatbot pour poser une question contextuelle/i,
      })
      .first()
      .click();
    // Panel must be visible (dialog role).
    await expect(page.getByRole('dialog')).toBeVisible();
    await expect(page.getByText(/Sentinel · XAU\/USD/i)).toBeVisible();
  });

  test('clicking a suggested question shows its scripted reply', async ({ page }) => {
    await page.goto('/#multi-marche');
    await page
      .getByRole('button', {
        name: /Ouvrir le chatbot pour poser une question contextuelle/i,
      })
      .first()
      .click();
    // Suggested question (5.D — descriptive, no synthetic score).
    await page
      .getByRole('button', { name: /Qu'est-ce que cette lecture me dit/i })
      .click();
    // Scripted reply describes the structure in plain language.
    await expect(
      page.getByText(/Cette lecture décrit un marché plutôt haussier/i),
    ).toBeVisible();
  });

  test('pedagogical refusal on "should I buy?"', async ({ page }) => {
    await page.goto('/#multi-marche');
    await page
      .getByRole('button', {
        name: /Ouvrir le chatbot pour poser une question contextuelle/i,
      })
      .first()
      .click();
    await page.getByRole('button', { name: /Donc je dois acheter/i }).click();
    // Refusal MUST start with "Non" — compliance UE 2024/2811.
    await expect(
      page.getByText(/Non, je ne donne aucune instruction/i),
    ).toBeVisible();
  });

  test('free-text input submits to the backend and renders the reply', async ({ page }) => {
    // Chantier 5.A: free-text now goes to POST /api/chatbot/message (backend
    // FastAPI, 3 defence layers). We mock the backend so the test is
    // deterministic and needs no running FastAPI. The 503-fallback and the
    // blocked_reason paths are covered in chatbot-backend-integration.spec.ts.
    await page.route('**/api/chatbot/message', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          content: 'En bref : XAUUSD H1 consolide, pas d’événement HIGH imminent.',
          blocked_reason: null,
          tool_calls_made: [],
        }),
      }),
    );

    await page.goto('/#multi-marche');
    await page
      .getByRole('button', {
        name: /Ouvrir le chatbot pour poser une question contextuelle/i,
      })
      .first()
      .click();
    const input = page.getByRole('textbox', {
      name: /Question libre pour Sentinel/i,
    });
    await expect(input).toBeVisible();
    await input.fill('Bonjour, en bref ?');
    await page.getByRole('button', { name: /Envoyer la question/i }).click();
    await expect(
      page.getByText(/En bref : XAUUSD H1 consolide/i),
    ).toBeVisible({ timeout: 10_000 });
  });
});
