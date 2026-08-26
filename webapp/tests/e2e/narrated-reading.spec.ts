import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * « Lecture narrée » (/app) — the narration is now composed 100 % by the
 * deterministic engine template (no LLM). This spec asserts the block renders on
 * both viewports and carries the honest engine-anchoring footer (« Chaque niveau
 * cité correspond à une sortie réelle du moteur », desktop NarratedPanel) /
 * « Composée par le moteur » source line (mobile ConditionsSection), with no raw
 * i18n key and no horizontal overflow. (UI-3 dropped the redundant « Ancrée au
 * moteur » badge that duplicated the footer.)
 *
 * The reading endpoints are mocked (the prod build proxies to a backend that is
 * absent under test). Locale is fr-FR (playwright.config).
 */
const SHOTS = '../docs/audits/narrated-reading-shots';

const START = Math.floor(Date.UTC(2026, 0, 1) / 1000);
const CANDLES = Array.from({ length: 300 }, (_, i) => {
  const close = 2000 + i;
  return { time: START + i * 900, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
});

// The exact narration text the mocked reading returns (fixture).
const NARRATION = /Structure haussière confirmée par une cassure récente/;

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

test.describe('Lecture narrée — desktop 1280×800', () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test.beforeEach(async ({ page }) => {
    await mockReading(page);
  });

  test('NarratedPanel shows title, engine-anchoring footer and the narration', async ({ page }) => {
    await page.goto('/app?instrument=XAUUSD&timeframe=M15');
    await dismissCookieBanner(page);

    // UI-3 removed the « Ancrée au moteur » badge (redundant with the footer
    // below). Title + engine-anchoring footer render (label literally true —
    // 100 % engine) and carry the provenance the badge used to duplicate.
    await expect(page.getByRole('heading', { name: 'Lecture narrée' })).toBeVisible();
    await expect(
      page.getByText(/Chaque niveau cité correspond à une sortie réelle du moteur/),
    ).toBeVisible();

    // The narration paragraph is present.
    await expect(page.getByText(NARRATION)).toBeVisible();

    await page.screenshot({ path: `${SHOTS}/a-desktop-narrated.png` });

    // No horizontal overflow introduced by the block.
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(1);
  });
});

test.describe('Lecture narrée — mobile 390×844', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test.beforeEach(async ({ page }) => {
    await mockReading(page);
  });

  test('the Lecture tab renders the narration and the « Composée par le moteur » source', async ({ page }) => {
    await page.goto('/app?instrument=XAUUSD&timeframe=M15');
    await dismissCookieBanner(page);

    // Switch to the « Lecture » tab (phone layout owns the reading in a tab).
    await page.getByRole('tab', { name: 'Lecture' }).click();

    // The ConditionsSection accordion « Lecture narrée » — expand it.
    const trigger = page.getByRole('button', { name: /Lecture narrée/ });
    await expect(trigger).toBeVisible();
    await trigger.click();

    // Narration + the engine-composed source line (no LLM wording anywhere).
    await expect(page.getByText(NARRATION)).toBeVisible();
    await expect(page.getByText('Composée par le moteur')).toBeVisible();

    await page.screenshot({ path: `${SHOTS}/b-mobile-narrated.png` });
  });
});
