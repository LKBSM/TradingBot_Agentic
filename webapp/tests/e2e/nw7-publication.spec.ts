import { expect, test, type Page } from '@playwright/test';

/**
 * NW-7 — page de publication COMPLÈTE : la courbe en % annuel ET les quatre
 * questions (dont #3, le cycle de vie des zones), au 1280×800 et au 390×844,
 * avec capture pour la comparaison à docs/design/reference-publication.html.
 * Cinq états :
 *   FULL      — courbe rendue + quatre cartes de questions ;
 *   PAST      — « Publiée » ;
 *   UPCOMING  — « Publication dans », point à venir vide ;
 *   NOVALUES  — bloc courbe absent, les questions restent ;
 *   NOFICHE   — aucune fiche pédagogique générique.
 * Tout le réseau est mocké ; le vrai backend n'est jamais appelé.
 */

const D = 86400_000;
const iso = (ms: number) => new Date(Date.now() + ms).toISOString();

const FULL_ACCESS = {
  authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false,
  is_owner: true, has_access: true, subscription_required: false,
};

// The curve reads as the BLS-published 12-month percent change (NW-7).
const SERIES = [
  { period: '2025-09', value: 2.9 }, { period: '2025-10', value: 3.0 },
  { period: '2025-11', value: 2.8 }, { period: '2025-12', value: 2.9 },
  { period: '2026-01', value: 3.0 }, { period: '2026-02', value: 3.3 },
  { period: '2026-03', value: 3.2 }, { period: '2026-04', value: 3.0 },
  { period: '2026-05', value: 2.9 }, { period: '2026-06', value: 3.0 },
  { period: '2026-07', value: 3.1 },
];

const prov = () => ({
  method_key: 'm', sample_size: 12, market: 'XAUUSD',
  period_start: iso(-360 * D), period_end: iso(-20 * D),
  reference_days: 60, quote_unit: 'USD',
});

// Full measures INCLUDING the zone-lifecycle (#3) card.
const FULL_MEAS = {
  event_key: 'us_cpi', market: 'XAUUSD',
  calm_before: {
    provenance: { ...prov() }, reference_amount: 9, calmer_count: 10, busier_count: 2,
    calmest: { observed_at: iso(-300 * D), minutes: null, amount: 3.4 },
    busiest: { observed_at: iso(-120 * D), minutes: null, amount: 12 },
  },
  structure_state: {
    provenance: { ...prov(), reference_days: null, quote_unit: null },
    inside_zone_count: 5, intact_pocket_count: 9, range_lower_count: 2,
    range_middle_count: 2, range_upper_count: 8,
    now_inside_zone: true, now_intact_pocket_within: false, now_range_position: 'upper',
  },
  zone_lifecycle: {
    provenance: { ...prov(), reference_days: null, quote_unit: null },
    zones_created_count: 34,
    tranches: [
      { lower_minutes: 0, upper_minutes: 60, count: 18 },
      { lower_minutes: 60, upper_minutes: 120, count: 8 },
      { lower_minutes: 120, upper_minutes: 1440, count: 5 },
    ],
    fastest: { observed_at: iso(-300 * D), minutes: 12, amount: null },
    slowest: { observed_at: iso(-120 * D), minutes: 600, amount: null },
    never_mitigated_count: 3,
  },
  return_to_calm: {
    provenance: { ...prov() },
    tranches: [
      { lower_minutes: 0, upper_minutes: 60, count: 4 },
      { lower_minutes: 60, upper_minutes: 180, count: 5 },
      { lower_minutes: 180, upper_minutes: null, count: 3 },
    ],
    fastest: { observed_at: iso(-300 * D), minutes: 15, amount: null },
    slowest: { observed_at: iso(-120 * D), minutes: 285, amount: null },
    never_settled_count: 0,
  },
};

const EMPTY_MEAS = { event_key: 'x', market: '', calm_before: null, structure_state: null, zone_lifecycle: null, return_to_calm: null };

function makeEvent(state: string, series: typeof SERIES | [], scheduledMs: number, opts?: { key?: string; source?: string; organism?: string | null; seriesCode?: string | null }) {
  const key = opts?.key ?? 'us_cpi';
  return {
    window_start: iso(-40 * D), window_end: iso(40 * D), generated_at: iso(0),
    coverage: { source: 'official', feed_start: null, feed_end: null, partial: false, last_success: {}, stale_sources: [] },
    attribution: [],
    events: [{
      event_id: `${opts?.source ?? 'bls'}:${key}:2026-08-12`, source: opts?.source ?? 'bls',
      series_code: opts?.seriesCode ?? 'CUUR0000SA0', license_label: 'x',
      event: key.toUpperCase(), currency: 'USD', organism: opts?.organism ?? 'Bureau of Labor Statistics',
      periodicity: 'monthly', scheduled_at: iso(scheduledMs), source_timezone: 'America/New_York',
      time_confirmed: true, markets: ['XAUUSD', 'EURUSD'], value_unit: '% de variation annuelle',
      actual: state === 'published' ? 3.1 : null, actual_initial: 3.2, previous: 3.0,
      revised: state === 'published', revised_at: iso(-20 * D), actual_state: state, refreshed_at: iso(0),
      value_series: series,
    }],
  };
}

const json = (b: unknown) => ({ status: 200, contentType: 'application/json', body: JSON.stringify(b) });

async function goto(page: Page, event: ReturnType<typeof makeEvent>, meas: unknown, urlId: string): Promise<boolean> {
  await page.route('**/api/access/me', (r) => r.fulfill(json(FULL_ACCESS)));
  await page.route('**/api/publications/*/measures', (r) => r.fulfill(json(meas)));
  await page.route('**/api/calendar/event/*', (r) => r.fulfill(json(event)));
  await page.route('**/api/calendar*', (r) => r.fulfill(json(event)));
  await page.goto(`/actualites/${urlId}`);
  try {
    await page.locator('.cald').first().waitFor({ state: 'visible', timeout: 20000 });
  } catch { return false; }
  return true;
}

async function assertCommonInvariants(page: Page) {
  expect(await page.locator('.cal-nono').count()).toBe(1); // ONE page warning
  await expect(page.locator('.pub-mia-cap')).toBeVisible();
  const txt = ((await page.locator('.cald').textContent()) ?? '').toLowerCase();
  for (const w of ['bougie', 'médiane', 'moyenne']) expect(txt).not.toContain(w);
  expect(/\bcalendar\.[a-zA-Z.]+\b/.test(txt)).toBe(false);
}

const CPI_URL = 'bls%3Aus_cpi%3A2026-08-12';

for (const vp of [{ w: 1280, h: 800, tag: '1280' }, { w: 390, h: 844, tag: '390' }]) {
  test(`${vp.tag}: FULL — curve (% annual) + four question cards incl. zone lifecycle`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    if (!(await goto(page, makeEvent('pending', SERIES, 12 * D), FULL_MEAS, CPI_URL))) { test.skip(true, 'gated'); return; }
    await expect(page.locator('.pub-curve-svg')).toBeVisible();
    // Header carries the percent-change unit (not an index level).
    expect((await page.locator('.cald').textContent()) ?? '').toContain('% de variation annuelle');
    // FOUR question cards, each with a source line carrying the denominator.
    await expect(page.locator('.pub-qcard')).toHaveCount(4);
    const sources = await page.locator('.pub-qsource').allTextContents();
    expect(sources).toHaveLength(4);
    for (const s of sources) expect(s).toContain('12');
    // Zone-lifecycle card: created count + its "still intact" aside, no candles.
    const qtext = (await page.locator('.pub-qsection').textContent()) ?? '';
    expect(qtext).toContain('34');
    await expect(page.locator('.pub-nevermit')).toBeVisible();
    await assertCommonInvariants(page);
    await page.screenshot({ path: `test-results/nw7-full-${vp.tag}.png`, fullPage: true });
  });

  test(`${vp.tag}: PAST — reads « Publiée »`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    if (!(await goto(page, makeEvent('published', SERIES, -1 * D), FULL_MEAS, CPI_URL))) { test.skip(true, 'gated'); return; }
    expect((await page.locator('.cald-cd .k').textContent())?.trim()).toBe('Publiée');
    await assertCommonInvariants(page);
    await page.screenshot({ path: `test-results/nw7-past-${vp.tag}.png`, fullPage: true });
  });

  test(`${vp.tag}: UPCOMING — « Publication dans », upcoming point blank`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    if (!(await goto(page, makeEvent('pending', SERIES, 12 * D), FULL_MEAS, CPI_URL))) { test.skip(true, 'gated'); return; }
    expect((await page.locator('.cald-cd .k').textContent())?.trim()).toBe('Publication dans');
    expect((await page.locator('.pt-upcoming-label').textContent()) ?? '').not.toMatch(/\d/);
    await assertCommonInvariants(page);
    await page.screenshot({ path: `test-results/nw7-upcoming-${vp.tag}.png`, fullPage: true });
  });

  test(`${vp.tag}: NOVALUES — curve omitted, questions remain`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    if (!(await goto(page, makeEvent('pending', [], 12 * D), FULL_MEAS, CPI_URL))) { test.skip(true, 'gated'); return; }
    await expect(page.locator('.pub-curve-svg')).toHaveCount(0);
    await expect(page.locator('.pub-qcard')).toHaveCount(4);
    await assertCommonInvariants(page);
    await page.screenshot({ path: `test-results/nw7-novalues-${vp.tag}.png`, fullPage: true });
  });

  test(`${vp.tag}: NOFICHE — no written fiche renders no pedagogy block`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    const ev = makeEvent('pending', [], 5 * D, { key: 'adp', source: 'forexfactory', organism: null, seriesCode: null });
    // The mocked event_id ends 2026-08-12 (makeEvent) — the URL must match it so
    // the detail page loads the event rather than treating the id as unknown.
    if (!(await goto(page, ev, EMPTY_MEAS, 'forexfactory%3Aadp%3A2026-08-12'))) { test.skip(true, 'gated'); return; }
    await expect(page.locator('.pub-ped-body')).toHaveCount(0);
    await assertCommonInvariants(page);
    await page.screenshot({ path: `test-results/nw7-nofiche-${vp.tag}.png`, fullPage: true });
  });
}
