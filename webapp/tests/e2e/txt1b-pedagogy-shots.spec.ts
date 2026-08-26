import { test, type Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

/** TXT-1b — captures of the pedagogy fiche collapsed by default and expanded. */

const OUT = path.join('..', 'docs', 'audits', 'txt1-shots', 'pedagogy-fold');
const D = 86400_000;
const iso = (ms: number) => new Date(Date.now() + ms).toISOString();
const FULL_ACCESS = {
  authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false,
  is_owner: true, has_access: true, subscription_required: false,
};
const EVENT = {
  window_start: iso(-40 * D), window_end: iso(40 * D), generated_at: iso(0),
  coverage: { source: 'official', feed_start: null, feed_end: null, partial: false, last_success: {}, stale_sources: [] },
  attribution: [],
  events: [{
    event_id: 'bls:us_cpi:2026-08-12', source: 'bls', series_code: 'CUUR0000SA0', license_label: 'x',
    event: 'US_CPI', currency: 'USD', organism: 'Bureau of Labor Statistics', periodicity: 'monthly',
    scheduled_at: iso(12 * D), source_timezone: 'America/New_York', time_confirmed: true,
    markets: ['XAUUSD', 'EURUSD'], value_unit: '% de variation annuelle',
    actual: null, actual_initial: 3.2, previous: 3.0, revised: false, revised_at: iso(-20 * D),
    actual_state: 'pending', refreshed_at: iso(0), value_series: [],
  }],
};
const MEAS = { event_key: 'us_cpi', market: '', calm_before: null, structure_state: null, zone_lifecycle: null, return_to_calm: null };
const json = (b: unknown) => ({ status: 200, contentType: 'application/json', body: JSON.stringify(b) });
const CPI_URL = 'bls%3Aus_cpi%3A2026-08-12';

async function mock(page: Page) {
  await page.route('**/api/access/me', (r) => r.fulfill(json(FULL_ACCESS)));
  await page.route('**/api/publications/*/measures', (r) => r.fulfill(json(MEAS)));
  await page.route('**/api/calendar/event/*', (r) => r.fulfill(json(EVENT)));
  await page.route('**/api/calendar*', (r) => r.fulfill(json(EVENT)));
}

test('pedagogy fiche collapsed then expanded (fr)', async ({ page }) => {
  fs.mkdirSync(OUT, { recursive: true });
  await mock(page);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto(`/fr/actualites/${CPI_URL}`);
  await page.locator('.pub-ped').first().waitFor({ state: 'visible', timeout: 30000 });
  await page.locator('.pub-ped').scrollIntoViewIfNeeded();
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT, 'fr-collapsed.png'), fullPage: true });
  await page.locator('.pub-ped summary').click();
  await page.locator('.pub-ped-body').waitFor({ state: 'visible' });
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT, 'fr-expanded.png'), fullPage: true });
});
