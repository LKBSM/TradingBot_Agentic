import { defineConfig, devices } from '@playwright/test';

/**
 * PAY-3 (G1) — configuration du test VIVANT (carte de test réelle).
 *
 * Séparée de `playwright.config.ts` pour deux raisons :
 *   · la suite e2e normale ne doit jamais toucher au réseau Stripe ;
 *   · ce test n'a PAS de `webServer` — le backend et le front sont démarrés par
 *     le workflow, avec `stripe listen` qui renvoie les webhooks, parce qu'un
 *     serveur lancé par Playwright ne recevrait pas ces webhooks.
 *
 * Aucun `retries` : une réussite au deuxième essai masquerait exactement la
 * latence anormale de webhook qu'on cherche à détecter.
 */
export default defineConfig({
  testDir: './tests/live',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  use: {
    baseURL: process.env.PAY3_APP_BASE ?? 'http://localhost:3000',
    trace: 'retain-on-failure',
    locale: 'fr-FR',
  },
  projects: [{ name: 'chromium-desktop', use: { ...devices['Desktop Chrome'] } }],
});
