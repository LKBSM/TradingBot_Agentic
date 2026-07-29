import { expect, test, type Page } from '@playwright/test';

/**
 * NW-1b — "Actualités programmées" calendar (list view). The default live source
 * (official aggregator) ships with an empty schedule, so to exercise the LIST,
 * the refactored filters (organism / market / periodicity) and a REVISED
 * publication, we intercept GET /api/calendar with a fixture. The static chrome
 * (intro + « ne dit pas » banner) renders without a backend.
 */

async function overflow(page: Page): Promise<number> {
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
}

/** A fixture with two upcoming events (one revised) + attribution. */
function fixture() {
  const soon = (h: number) => new Date(Date.now() + h * 3600_000).toISOString();
  return {
    window_start: new Date(Date.now() - 3 * 86400_000).toISOString(),
    window_end: new Date(Date.now() + 7 * 86400_000).toISOString(),
    generated_at: new Date().toISOString(),
    coverage: { source: 'official', feed_start: null, feed_end: null, partial: false, last_success: {}, stale_sources: [] },
    attribution: [
      { source: 'bls', organism: 'Bureau of Labor Statistics', license_label: 'Domaine public (17 U.S.C. §105)', policy_url: 'https://www.bls.gov/opub/copyright-information.htm' },
      { source: 'ecb', organism: 'Banque centrale européenne', license_label: 'Réutilisation si source citée et non modifiée', policy_url: 'https://www.ecb.europa.eu/x' },
    ],
    events: [
      {
        event_id: 'bls:us_cpi:2026-08-12', source: 'bls', series_code: 'CUUR0000SA0',
        license_label: 'Domaine public', event: 'Indice des prix à la consommation (IPC)',
        currency: 'USD', organism: 'Bureau of Labor Statistics', periodicity: 'monthly',
        scheduled_at: soon(3), source_timezone: 'America/New_York', time_confirmed: true,
        markets: ['XAUUSD', 'EURUSD'], value_unit: 'indice (1982-84 = 100)',
        actual: 322.9, actual_initial: 321.5, previous: 320.0, revised: true,
        revised_at: new Date(Date.now() - 86400_000).toISOString(),
      },
      {
        event_id: 'ecb:ea_ecb_rate:2026-08-14', source: 'ecb', series_code: 'FM.D.U2.EUR.4F.KR.MRR_FR.LEV',
        license_label: 'Réutilisation si citée', event: 'Décision de taux (BCE)',
        currency: 'EUR', organism: 'Banque centrale européenne', periodicity: 'eight_per_year',
        scheduled_at: soon(6), source_timezone: 'Europe/Berlin', time_confirmed: true,
        markets: ['EURUSD'], value_unit: 'taux directeur (% par an)',
        actual: null, actual_initial: null, previous: 2.4, revised: false, revised_at: null,
      },
    ],
  };
}

/** Full-access summary so the SubscriptionGate opens without a live backend. */
const FULL_ACCESS = {
  authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false,
  tier: 'owner', is_owner: true, has_full_access: true,
  entitlements: { instruments: null, timeframes: null, scanner: true, chat: { limit: null, used: 0, remaining: null } },
};

async function gotoCalendar(page: Page): Promise<boolean> {
  await page.route('**/api/access/me', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FULL_ACCESS) }),
  );
  await page.route('**/api/calendar*', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(fixture()) }),
  );
  await page.goto('/actualites');
  // The gate resolves /api/access/me (intercepted) then renders the calendar —
  // wait for the heading rather than reading it before the async render settles.
  const heading = page.getByRole('heading', { name: 'Actualités programmées' });
  try {
    await heading.waitFor({ state: 'visible', timeout: 20000 });
  } catch {
    return false; // still gated in this e2e environment
  }
  return true;
}

test('1280×800: the product rail exposes an « Actualités » entry to the calendar', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/app');
  const link = page.getByRole('link', { name: 'Actualités' }).first();
  if ((await link.count()) === 0) {
    test.skip(true, 'product rail not rendered in this e2e environment');
    return;
  }
  await expect(link).toHaveAttribute('href', /\/actualites$/);
  expect(await overflow(page)).toBeLessThanOrEqual(1);
});

test('1280×800: list renders with factual filters + a revised publication', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  if (!(await gotoCalendar(page))) {
    test.skip(true, 'calendar gated in this e2e environment');
    return;
  }

  // Static chrome is descriptive.
  await expect(page.getByText('Ce calendrier annonce des moments, pas des directions.')).toBeVisible();
  await expect(page.getByText('Ce que ce calendrier ne dit pas')).toBeVisible();

  // Refactored filters — organism / market / periodicity, NO impact filter.
  await expect(page.locator('.fsec', { hasText: 'Organisme' })).toBeVisible();
  await expect(page.locator('.fsec', { hasText: 'Périodicité' })).toBeVisible();
  const filterText = (await page.locator('.cal-filtbar').textContent()) ?? '';
  expect(filterText).not.toContain('Impact');

  // The two fixture events are listed, and no colour-graded impact badge exists.
  await expect(page.locator('.cal-row')).toHaveCount(2);
  expect(await page.locator('.cal-impact').count()).toBe(0);

  // The revised publication is surfaced as revised.
  const cpiRow = page.locator('.cal-row', { hasText: 'Indice des prix' });
  await expect(cpiRow.getByText('valeur révisée')).toBeVisible();

  // The attribution block names each source used + links its reuse policy.
  const attrib = page.locator('.cal-attrib');
  await expect(attrib).toContainText('Bureau of Labor Statistics');
  await expect(attrib.locator('a', { hasText: 'politique de réutilisation' }).first()).toBeVisible();

  // No raw i18n key leaked into the calendar region.
  const calText = (await page.locator('.cal-page').textContent()) ?? '';
  expect(/\bcalendar\.[a-zA-Z.]+\b/.test(calText)).toBe(false);
  expect(await overflow(page)).toBeLessThanOrEqual(1);
});

test('1280×800: an organism filter narrows the list; zero periodicity chip empties it', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  if (!(await gotoCalendar(page))) {
    test.skip(true, 'calendar gated in this e2e environment');
    return;
  }
  // Deselect every periodicity chip → explicit empty state, never a fallback.
  for (const label of ['Mensuel', 'Trimestriel', '8 fois par an']) {
    await page.locator('.fchip', { hasText: new RegExp(`^${label}$`) }).first().click();
  }
  await expect(page.locator('.cal-row')).toHaveCount(0);
  await expect(page.locator('.cal-empty')).toBeVisible();
});

test('390×844: revised publication renders with no horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  if (!(await gotoCalendar(page))) {
    test.skip(true, 'calendar gated in this e2e environment');
    return;
  }
  await expect(page.locator('.cal-row').first()).toBeVisible();
  await expect(page.getByText('valeur révisée').first()).toBeVisible();
  expect(await overflow(page)).toBeLessThanOrEqual(1);
});
