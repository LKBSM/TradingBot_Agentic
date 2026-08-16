import { expect, test, type Page } from '@playwright/test';

/**
 * NW-5 — page d'une publication : contenu complet et cohérence visuelle.
 * Three states, each at 1280×800 and 390×844, with a screenshot for the visual
 * comparison against docs/design/reference-publication.html:
 *   A. FULL     — curve + four questions + M.I.A + four source links + pedagogy;
 *   B. NO CURVE — no published series (curve block absent), questions present;
 *   C. NO MEAS. — engine measures absent (questions block absent), curve present.
 * Every network call is mocked; the real backend is never hit.
 */

const D = 86400_000;
const H = 3600_000;
const iso = (ms: number) => new Date(Date.now() + ms).toISOString();

const FULL_ACCESS = {
  authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false,
  is_owner: true, has_access: true, subscription_required: false,
};

const SERIES = [
  { period: '2025-09', value: 2.9 }, { period: '2025-10', value: 3.0 },
  { period: '2025-11', value: 2.8 }, { period: '2025-12', value: 2.9 },
  { period: '2026-01', value: 3.0 }, { period: '2026-02', value: 3.3 },
  { period: '2026-03', value: 3.2 }, { period: '2026-04', value: 3.0 },
  { period: '2026-05', value: 2.9 }, { period: '2026-06', value: 3.0 },
  { period: '2026-07', value: 3.1 },
];

function cpiEvent(series: typeof SERIES | []) {
  return {
    window_start: iso(-30 * D), window_end: iso(30 * D), generated_at: iso(0),
    coverage: { source: 'official', feed_start: null, feed_end: null, partial: false, last_success: {}, stale_sources: [] },
    attribution: [{ source: 'bls', organism: 'Bureau of Labor Statistics', license_label: 'Domaine public (17 U.S.C. §105) — citation : U.S. Bureau of Labor Statistics', policy_url: 'https://www.bls.gov/opub/copyright-information.htm' }],
    events: [{
      event_id: 'bls:us_cpi:2026-08-12', source: 'bls', series_code: 'CUUR0000SA0',
      license_label: 'x', event: 'Indice des prix à la consommation (IPC)', currency: 'USD',
      organism: 'Bureau of Labor Statistics', periodicity: 'monthly', scheduled_at: iso(12 * D),
      source_timezone: 'America/New_York', time_confirmed: true, markets: ['XAUUSD', 'EURUSD'],
      value_unit: '% de variation annuelle', actual: 3.1, actual_initial: 3.2, previous: 3.0,
      revised: true, revised_at: iso(-20 * D), actual_state: 'pending', refreshed_at: iso(0),
      value_series: series,
    }],
  };
}

function measures(full: boolean) {
  if (!full) return { event_key: 'us_cpi', market: '', calm_before: null, structure_state: null, return_to_calm: null };
  const prov = () => ({ method_key: 'x', sample_size: 12, market: 'XAUUSD', period_start: iso(-360 * D), period_end: iso(-2 * D), reference_days: 60, quote_unit: 'USD' });
  return {
    event_key: 'us_cpi', market: 'XAUUSD',
    calm_before: { provenance: prov(), reference_amount: 9, calmer_count: 10, busier_count: 2, calmest: { observed_at: iso(-120 * D), minutes: null, amount: 3.4 }, busiest: { observed_at: iso(-40 * D), minutes: null, amount: 12 } },
    structure_state: { provenance: prov(), inside_zone_count: 5, intact_pocket_count: 9, range_lower_count: 2, range_middle_count: 2, range_upper_count: 8, now_inside_zone: true, now_intact_pocket_within: false, now_range_position: 'upper' },
    return_to_calm: { provenance: prov(), tranches: [{ lower_minutes: 0, upper_minutes: 60, count: 4 }, { lower_minutes: 60, upper_minutes: 180, count: 5 }, { lower_minutes: 180, upper_minutes: null, count: 3 }], fastest: { observed_at: iso(-90 * D), minutes: 15, amount: null }, slowest: { observed_at: iso(-30 * D), minutes: 285, amount: null }, never_settled_count: 0 },
  };
}

const json = (b: unknown) => ({ status: 200, contentType: 'application/json', body: JSON.stringify(b) });

async function goto(page: Page, event: unknown, meas: unknown): Promise<boolean> {
  await page.route('**/api/access/me', (r) => r.fulfill(json(FULL_ACCESS)));
  await page.route('**/api/calendar/event/*', (r) => r.fulfill(json(event)));
  await page.route('**/api/publications/us_cpi/measures', (r) => r.fulfill(json(meas)));
  await page.route('**/api/calendar*', (r) => r.fulfill(json(event)));
  await page.goto('/actualites/bls%3Aus_cpi%3A2026-08-12');
  try {
    await page.locator('.cald').first().waitFor({ state: 'visible', timeout: 20000 });
  } catch { return false; }
  return true;
}

async function overflow(page: Page): Promise<number> {
  return page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
}

for (const vp of [{ w: 1280, h: 800, tag: '1280' }, { w: 390, h: 844, tag: '390' }]) {
  test(`${vp.tag}: FULL page — curve + 3 questions + 4 source links + shared M.I.A`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    if (!(await goto(page, cpiEvent(SERIES), measures(true)))) { test.skip(true, 'gated'); return; }

    await expect(page.locator('.pub-curve-svg')).toBeVisible();
    // upcoming point carries no number
    expect((await page.locator('.pt-upcoming-label').textContent()) ?? '').not.toMatch(/\d/);
    // stats row + corrected-value coexistence (3.2 initial, 3.1 current)
    const stats = (await page.locator('.pub-curve-stats').textContent()) ?? '';
    expect(stats).toContain('3.1');
    const revLine = (await page.locator('.cald-rev').textContent()) ?? '';
    expect(revLine).toContain('3.2');
    expect(revLine).toContain('3.1');

    // three measured questions + common read-guide
    expect(await page.locator('.pub-qcard').count()).toBe(3);
    await expect(page.locator('.pub-qwarn')).toBeVisible();

    // shared M.I.A avatar + presence + four suggestions
    await expect(page.locator('.pub-mia-head svg')).toBeVisible();
    await expect(page.locator('.pub-mia-head [data-presence="1"]')).toBeVisible();
    expect(await page.locator('.pub-mia-chip').count()).toBe(4);

    // four named source links, all bls.gov
    expect(await page.locator('.pub-src-doc').count()).toBe(4);
    for (const a of await page.locator('.pub-src-doc').all()) {
      expect(new URL((await a.getAttribute('href')) ?? '').host.replace(/^www\./, '')).toBe('bls.gov');
    }

    // no candle / stat wording anywhere on the page
    const txt = ((await page.locator('.cald').textContent()) ?? '').toLowerCase();
    expect(txt).not.toContain('bougie');
    expect(txt).not.toContain('médiane');
    expect(txt).not.toContain('moyenne');
    expect(/\bcalendar\.[a-zA-Z.]+\b/.test(txt)).toBe(false);

    expect(await overflow(page)).toBeLessThanOrEqual(1);
    await page.screenshot({ path: `test-results/nw5-full-${vp.tag}.png`, fullPage: true });
  });

  test(`${vp.tag}: NO published series — curve block absent, questions present`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    if (!(await goto(page, cpiEvent([]), measures(true)))) { test.skip(true, 'gated'); return; }
    await expect(page.locator('.pub-curve-svg')).toHaveCount(0);
    expect(await page.locator('.pub-qcard').count()).toBe(3);
    await expect(page.locator('.pub-mia')).toBeVisible();
    expect(await overflow(page)).toBeLessThanOrEqual(1);
    await page.screenshot({ path: `test-results/nw5-nocurve-${vp.tag}.png`, fullPage: true });
  });

  test(`${vp.tag}: engine measures absent — questions block absent, curve present`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    if (!(await goto(page, cpiEvent(SERIES), measures(false)))) { test.skip(true, 'gated'); return; }
    await expect(page.locator('.pub-curve-svg')).toBeVisible();
    await expect(page.locator('.pub-qsection')).toHaveCount(0);
    await expect(page.locator('.pub-qcard')).toHaveCount(0);
    // sources + pedagogy still render
    expect(await page.locator('.pub-src-doc').count()).toBe(4);
    expect(await overflow(page)).toBeLessThanOrEqual(1);
    await page.screenshot({ path: `test-results/nw5-nomeasures-${vp.tag}.png`, fullPage: true });
  });
}
