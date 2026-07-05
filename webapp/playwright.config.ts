import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright config for end-to-end tests. Spins up `next dev` automatically.
 *
 * Run once after `npm install`:
 *
 *   npx playwright install chromium
 *
 * Then:
 *
 *   npm run test:e2e          # headless run
 *   npm run test:e2e:ui       # interactive UI mode
 */
// E2E_BASE_URL pilote à la fois le navigateur et le health-check du serveur —
// utile pour tourner sur un autre port quand 3000 est occupé (PORT est lu par
// `next start`/`next dev`) : PORT=3100 E2E_BASE_URL=http://localhost:3100.
const BASE_URL = process.env.E2E_BASE_URL ?? 'http://localhost:3000';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: BASE_URL,
    trace: 'on-first-retry',
    locale: 'fr-FR',
  },
  projects: [
    {
      name: 'chromium-desktop',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'mobile-iphone-12',
      // Le device « iPhone 12 » sélectionne webkit par défaut ; la CI (et la
      // convention locale documentée ci-dessus) n'installe que chromium →
      // chaque test mobile échouait au lancement du navigateur. On garde le
      // viewport/UA iPhone mais émulé sous chromium.
      use: { ...devices['iPhone 12'], browserName: 'chromium' },
    },
  ],
  webServer: {
    // En CI le workflow fait déjà `npm run build` : servir le build (start)
    // est plus rapide et plus fidèle à la prod que `next dev`.
    command: process.env.CI ? 'npm run start' : 'npm run dev',
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
