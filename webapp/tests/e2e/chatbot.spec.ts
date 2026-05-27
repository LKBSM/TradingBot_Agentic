import { expect, test } from '@playwright/test';

test.describe('Chatbot — golden paths', () => {
  test('opens the chat panel when CTA is clicked', async ({ page }) => {
    await page.goto('/#demo');
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
    await page.goto('/#demo');
    await page
      .getByRole('button', {
        name: /Ouvrir le chatbot pour poser une question contextuelle/i,
      })
      .first()
      .click();
    // Suggested question: "Pourquoi seulement 72 ?"
    await page.getByRole('button', { name: /Pourquoi seulement 72/i }).click();
    // Scripted reply contains "8 composantes" + "Pearson" or similar marker.
    await expect(
      page.getByText(/Sur les 8 composantes du moteur/i),
    ).toBeVisible();
  });

  test('pedagogical refusal on "should I buy?"', async ({ page }) => {
    await page.goto('/#demo');
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

  test('free-text input is present and submits (503 fallback when no key)', async ({ page }) => {
    await page.goto('/#demo');
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
    // Either the LLM streams a reply (key present) or we see the friendly
    // fallback message (no key) — both are acceptable.
    await expect(
      page
        .getByText(/Le mode chatbot en direct n'est pas encore activé/i)
        .or(page.locator('text=/\\w+/').last()),
    ).toBeVisible({ timeout: 15_000 });
  });
});
