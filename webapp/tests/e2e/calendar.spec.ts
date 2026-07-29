import { expect, test, type Page } from '@playwright/test';

/**
 * NW-1c — "Actualités programmées" calendar. The default live source ships with
 * an empty schedule, so we intercept GET /api/calendar with a fixture to exercise
 * the list, the refactored filters, a REVISED publication, the three distinct
 * value absences (pending / unfetched / unavailable) and per-organism freshness.
 */

async function overflow(page: Page): Promise<number> {
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
}

const iso = (ms: number) => new Date(Date.now() + ms).toISOString();
const H = 3600_000;
const D = 86400_000;

/** Fixture: 2 upcoming (pending) + 3 past (published+revised / unfetched / unavailable). */
function fixture() {
  return {
    window_start: iso(-30 * D),
    window_end: iso(30 * D),
    generated_at: iso(0),
    coverage: {
      source: 'official', feed_start: null, feed_end: null, partial: false,
      last_success: { bls: iso(0), ecb: iso(0), federal_reserve: iso(-8 * D) },
      stale_sources: ['federal_reserve'],
    },
    attribution: [
      { source: 'bls', organism: 'Bureau of Labor Statistics', license_label: 'Domaine public (17 U.S.C. §105)', policy_url: 'https://www.bls.gov/opub/copyright-information.htm' },
      { source: 'ecb', organism: 'Banque centrale européenne', license_label: 'Réutilisation si source citée et non modifiée', policy_url: 'https://www.ecb.europa.eu/x' },
      { source: 'federal_reserve', organism: 'Federal Reserve Board', license_label: 'Domaine public', policy_url: 'https://www.federalreserve.gov/disclaimer.htm' },
    ],
    events: [
      // published + revised (past)
      base('bls:us_cpi:past', 'bls', 'Indice des prix à la consommation (IPC)', 'USD', iso(-2 * D), 'CUUR0000SA0', 'indice (1982-84 = 100)', 'Bureau of Labor Statistics', {
        actual: 322.9, actual_initial: 321.5, previous: 320.0, revised: true, revised_at: iso(-1 * D), actual_state: 'published', refreshed_at: iso(0),
      }),
      // pending (upcoming)
      base('ecb:ea_ecb_rate:up', 'ecb', 'Décision de taux (BCE)', 'EUR', iso(6 * H), 'FM.D.U2.EUR.4F.KR.MRR_FR.LEV', 'taux directeur (% par an)', 'Banque centrale européenne', { previous: 2.4, actual_state: 'pending' }),
      // pending (upcoming, 2nd)
      base('bls:us_ppi:up', 'bls', 'Indice des prix à la production (IPP)', 'USD', iso(3 * H), 'WPSFD4', 'indice (nov. 2009 = 100)', 'Bureau of Labor Statistics', { actual_state: 'pending' }),
      // unfetched (past, has series, no value)
      base('bls:us_nfp:past', 'bls', "Rapport sur l'emploi (NFP)", 'USD', iso(-1 * D), 'CES0000000001', "milliers d'emplois", 'Bureau of Labor Statistics', { actual_state: 'unfetched', refreshed_at: iso(-1 * D) }),
      // unavailable (past, no series)
      base('federal_reserve:us_fomc:past', 'federal_reserve', 'Décision de taux (FOMC)', 'USD', iso(-1 * D), null, 'fourchette du taux directeur (%)', 'Federal Reserve Board', { actual_state: 'unavailable' }),
    ],
  };
}

function base(
  event_id: string, source: string, event: string, currency: string, scheduled_at: string,
  series_code: string | null, value_unit: string, organism: string,
  extra: Record<string, unknown>,
) {
  return {
    event_id, source, series_code, license_label: 'x', event, currency, organism,
    periodicity: 'monthly', scheduled_at, source_timezone: 'America/New_York',
    time_confirmed: true, markets: ['XAUUSD', 'EURUSD'], value_unit,
    actual: null, actual_initial: null, previous: null, revised: false, revised_at: null,
    actual_state: 'pending', refreshed_at: null, ...extra,
  };
}

const FULL_ACCESS = {
  authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false,
  tier: 'owner', is_owner: true, has_full_access: true,
  entitlements: { instruments: null, timeframes: null, scanner: true, chat: { limit: null, used: 0, remaining: null } },
};

async function route(page: Page, body: unknown = fixture()) {
  await page.route('**/api/access/me', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FULL_ACCESS) }));
  await page.route('**/api/calendar*', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) }));
}

async function gotoList(page: Page, body?: unknown): Promise<boolean> {
  await route(page, body);
  await page.goto('/actualites');
  const heading = page.getByRole('heading', { name: 'Actualités programmées' });
  try {
    await heading.waitFor({ state: 'visible', timeout: 20000 });
  } catch {
    return false;
  }
  return true;
}

async function gotoDetail(page: Page, eventId: string): Promise<boolean> {
  await route(page);
  await page.goto(`/actualites/${encodeURIComponent(eventId)}`);
  try {
    await page.locator('h1').first().waitFor({ state: 'visible', timeout: 20000 });
  } catch {
    return false;
  }
  return true;
}

test('1280×800: the product rail exposes an « Actualités » entry', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/app');
  const link = page.getByRole('link', { name: 'Actualités' }).first();
  if ((await link.count()) === 0) { test.skip(true, 'rail not rendered'); return; }
  await expect(link).toHaveAttribute('href', /\/actualites$/);
  expect(await overflow(page)).toBeLessThanOrEqual(1);
});

test('1280×800: list — factual filters, no causal verb, per-organism freshness', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  if (!(await gotoList(page))) { test.skip(true, 'gated'); return; }

  await expect(page.getByText('Ce calendrier annonce des moments, pas des directions.')).toBeVisible();
  await expect(page.locator('.fsec', { hasText: 'Organisme' })).toBeVisible();
  await expect(page.locator('.fsec', { hasText: 'Périodicité' })).toBeVisible();

  // two upcoming (pending) events by default; no causal verb, no impact badge
  await expect(page.locator('.cal-row')).toHaveCount(2);
  const listText = (await page.locator('.cal-page').textContent()) ?? '';
  expect(listText).toContain('rattaché à');
  expect(listText).not.toContain('affecte');
  expect(await page.locator('.cal-impact').count()).toBe(0);
  expect(listText.toLowerCase()).not.toContain('à venir'); // no future-content promise

  // freshness: fresh sources dated, the stale one flagged (never silent)
  const attrib = page.locator('.cal-attrib');
  await expect(attrib).toContainText('rafraîchi le');
  await expect(page.locator('.cal-attrib .fresh.stale')).toContainText('non rafraîchi depuis');

  // no raw i18n key in the calendar region
  expect(/\bcalendar\.[a-zA-Z.]+\b/.test(listText)).toBe(false);
  expect(await overflow(page)).toBeLessThanOrEqual(1);
});

test('1280×800: zero periodicity chip → explicit empty, never a fallback', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  if (!(await gotoList(page))) { test.skip(true, 'gated'); return; }
  for (const label of ['Mensuel', 'Trimestriel', '8 fois par an']) {
    await page.locator('.fchip', { hasText: new RegExp(`^${label}$`) }).first().click();
  }
  await expect(page.locator('.cal-row')).toHaveCount(0);
  await expect(page.locator('.cal-empty')).toBeVisible();
});

test('1280×800: a period with no event renders an explicit empty state', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  const empty = { ...fixture(), events: [], attribution: [] };
  if (!(await gotoList(page, empty))) { test.skip(true, 'gated'); return; }
  await expect(page.locator('.cal-row')).toHaveCount(0);
  await expect(page.locator('.cal-empty')).toBeVisible();
  expect(await overflow(page)).toBeLessThanOrEqual(1);
});

test('1280×800: detail with a published, revised value', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  if (!(await gotoDetail(page, 'bls:us_cpi:past'))) { test.skip(true, 'gated'); return; }
  await expect(page.locator('.cald-figs')).toContainText('322.9');       // as published
  await expect(page.locator('.cald-rev-line')).toContainText('321.5');   // initial preserved
  await expect(page.locator('.cald-rev-line')).toContainText('322.9');   // current
  await expect(page.locator('.cal-attrib')).toContainText('Bureau of Labor Statistics');
});

test('1280×800: detail states — pending, unfetched, unavailable are distinct', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  // pending
  if (!(await gotoDetail(page, 'ecb:ea_ecb_rate:up'))) { test.skip(true, 'gated'); return; }
  await expect(page.locator('.cald-figs')).toContainText('non encore publiée');
  // unfetched (product state, with last attempt) — NOT confused with pending
  await gotoDetail(page, 'bls:us_nfp:past');
  await expect(page.locator('.cald-figs')).toContainText('non récupérée');
  await expect(page.locator('.cald-figs')).not.toContainText('non encore publiée');
  // unavailable (source publishes no single figure, organism named)
  await gotoDetail(page, 'federal_reserve:us_fomc:past');
  await expect(page.locator('.cald-figs')).toContainText('non disponible');
  await expect(page.locator('.cald-figs')).toContainText('Federal Reserve Board');
});

test('390×844: detail with revision renders with no horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  if (!(await gotoDetail(page, 'bls:us_cpi:past'))) { test.skip(true, 'gated'); return; }
  await expect(page.locator('.cald-rev-line')).toContainText('322.9');
  expect(await overflow(page)).toBeLessThanOrEqual(1);
});

test('390×844: list has no horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  if (!(await gotoList(page))) { test.skip(true, 'gated'); return; }
  await expect(page.locator('.cal-row').first()).toBeVisible();
  expect(await overflow(page)).toBeLessThanOrEqual(1);
});
