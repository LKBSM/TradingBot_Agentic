import { expect, test, type Page } from '@playwright/test';

/**
 * NW-8 — la variation en évidence, le niveau en second, la phrase d'explication,
 * l'attribution. Au 1280×800 et au 390×844. Tout le réseau est mocké.
 * États : variation PUBLIÉE (index_change) · sans variation (niveau seul) ·
 * sans phrase d'explication rédigée. Le point à venir reste vide (niveau ET variation).
 */

const D = 86400_000;
const iso = (ms: number) => new Date(Date.now() + ms).toISOString();

const FULL_ACCESS = {
  authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false,
  tier: 'owner', is_owner: true, has_full_access: true,
  entitlements: { instruments: null, timeframes: null, scanner: true, chat: { limit: null, used: 0, remaining: null } },
};

// A value_series point. `change_mom` is the month-over-month variation and is
// null when it is not published/computed for that point (NW-8 fallback).
type SeriesPoint = { period: string; value: number; level: number; change_mom: number | null };

// index_change series: value = yoy %, level = raw index, change_mom = mo %.
const IDX_SERIES: SeriesPoint[] = [
  { period: '2025-09', value: 2.9, level: 330.1, change_mom: 0.2 },
  { period: '2025-10', value: 3.0, level: 331.0, change_mom: 0.3 },
  { period: '2025-11', value: 2.8, level: 331.4, change_mom: 0.1 },
  { period: '2026-05', value: 3.4, level: 335.1, change_mom: 0.5 },
  { period: '2026-06', value: 3.1, level: 333.9, change_mom: 0.3 },
];

const EMPTY_MEAS = { event_key: 'x', market: '', calm_before: null, structure_state: null, zone_lifecycle: null, return_to_calm: null };
const json = (b: unknown) => ({ status: 200, contentType: 'application/json', body: JSON.stringify(b) });

function makeEvent(o: {
  key: string; source: string; organism: string | null; seriesCode: string | null;
  valueUnit: string | null; variationKind: string | null; series: SeriesPoint[];
  variationPublished?: boolean;
}) {
  return {
    window_start: iso(-40 * D), window_end: iso(40 * D), generated_at: iso(0),
    coverage: { source: 'official', feed_start: null, feed_end: null, partial: false, last_success: {}, stale_sources: [] },
    attribution: [],
    events: [{
      event_id: `${o.source}:${o.key}:2026-08-12`, source: o.source, series_code: o.seriesCode,
      license_label: 'x', event: o.key.toUpperCase(), currency: 'USD', organism: o.organism,
      periodicity: 'monthly', scheduled_at: iso(12 * D), source_timezone: 'America/New_York',
      time_confirmed: true, markets: ['XAUUSD', 'EURUSD'], value_unit: o.valueUnit,
      variation_kind: o.variationKind, variation_published: o.variationPublished ?? true,
      actual: null, actual_initial: null, previous: null, revised: false, revised_at: null,
      actual_state: 'pending', refreshed_at: iso(0), value_series: o.series,
    }],
  };
}

async function goto(page: Page, event: ReturnType<typeof makeEvent>, urlId: string): Promise<boolean> {
  await page.route('**/api/access/me', (r) => r.fulfill(json(FULL_ACCESS)));
  await page.route('**/api/publications/*/measures', (r) => r.fulfill(json(EMPTY_MEAS)));
  await page.route('**/api/calendar/event/*', (r) => r.fulfill(json(event)));
  await page.route('**/api/calendar*', (r) => r.fulfill(json(event)));
  await page.goto(`/actualites/${urlId}`);
  try {
    await page.locator('.cald').first().waitFor({ state: 'visible', timeout: 20000 });
  } catch { return false; }
  return true;
}

const FORBIDDEN = ['accélère', 'ralentit', 'solide', 'décevant', 'au-dessus des attentes',
  'surprise', 'médiane', 'moyenne', 'bougie'];

async function assertNoForbiddenVocab(page: Page) {
  const txt = ((await page.locator('.cald').textContent()) ?? '').toLowerCase();
  for (const w of FORBIDDEN) expect(txt, `mot interdit: ${w}`).not.toContain(w);
  expect(/\bcalendar\.[a-zA-Z.]+\b/.test(txt)).toBe(false); // no raw i18n key
}

const CPI = { key: 'us_cpi', source: 'bls', organism: 'Bureau of Labor Statistics', seriesCode: 'CUUR0000SA0', valueUnit: 'indice (1982-84 = 100)', variationKind: 'index_change' };
const CPI_URL = 'bls%3Aus_cpi%3A2026-08-12';

for (const vp of [{ w: 1280, h: 800, tag: '1280' }, { w: 390, h: 844, tag: '390' }]) {
  test(`${vp.tag}: PUBLISHED variation — mo+yr in evidence, level small, attribution, blank upcoming`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    if (!(await goto(page, makeEvent({ ...CPI, series: IDX_SERIES }), CPI_URL))) { test.skip(true, 'gated'); return; }
    await expect(page.locator('.pub-curve-svg')).toBeVisible();
    // variation in evidence (headline block), raw level in the second plan
    await expect(page.locator('.pub-var-headline')).toBeVisible();
    await expect(page.locator('.pub-var-level')).toBeVisible();
    // attribution names the organism AND the series, marked published
    const attrib = (await page.locator('.pub-curve-attrib').textContent()) ?? '';
    expect(attrib).toContain('Bureau of Labor Statistics');
    expect(attrib).toContain('CUUR0000SA0');
    // a real explanation sentence for this publication
    await expect(page.locator('.pub-curve-explain')).toBeVisible();
    // the upcoming point carries NO number
    expect((await page.locator('.pt-upcoming-label').textContent()) ?? '').not.toMatch(/\d/);
    await assertNoForbiddenVocab(page);
    await page.screenshot({ path: `test-results/nw8-published-${vp.tag}.png`, fullPage: true });
  });

  test(`${vp.tag}: COMPUTED variation — monthly % in evidence, CALCULATED attribution`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    const AMT_SERIES: SeriesPoint[] = [
      { period: '2026-04', value: 0.3, level: 712000, change_mom: null },
      { period: '2026-05', value: 0.4, level: 716000, change_mom: null },
      { period: '2026-06', value: 0.6, level: 720000, change_mom: null },
    ];
    const ev = makeEvent({ key: 'us_retail_sales', source: 'census', organism: 'U.S. Census Bureau', seriesCode: 'MARTS-RSAFS', valueUnit: 'millions de dollars', variationKind: 'amount_change', variationPublished: false, series: AMT_SERIES });
    if (!(await goto(page, ev, 'census%3Aus_retail_sales%3A2026-08-12'))) { test.skip(true, 'gated'); return; }
    await expect(page.locator('.pub-var-headline')).toBeVisible();
    const attrib = ((await page.locator('.pub-curve-attrib').textContent()) ?? '').toLowerCase();
    expect(attrib).toContain('calculée'); // computed, not "publiée"
    await expect(page.locator('.pub-curve-explain')).toBeVisible();
    await assertNoForbiddenVocab(page);
    await page.screenshot({ path: `test-results/nw8-computed-${vp.tag}.png`, fullPage: true });
  });

  test(`${vp.tag}: NO variation — level only, no attribution/explain block`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    const ev = makeEvent({ key: 'us_fomc_minutes', source: 'federal_reserve', organism: 'Réserve fédérale', seriesCode: null, valueUnit: null, variationKind: null, series: [] });
    if (!(await goto(page, ev, 'federal_reserve%3Aus_fomc_minutes%3A2026-08-12'))) { test.skip(true, 'gated'); return; }
    await expect(page.locator('.pub-var-headline')).toHaveCount(0);
    await expect(page.locator('.pub-curve-attrib')).toHaveCount(0);
    await assertNoForbiddenVocab(page);
    await page.screenshot({ path: `test-results/nw8-novariation-${vp.tag}.png`, fullPage: true });
  });

  test(`${vp.tag}: variation but NO written sentence — no explanation block rendered`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    // A publication NOT whitelisted in CURVE_EXPLAIN → variation shown, NO sentence.
    const ev = makeEvent({ key: 'adp', source: 'forexfactory', organism: 'ADP', seriesCode: 'ADP-NEP', valueUnit: 'indice (2017 = 100)', variationKind: 'index_change', series: IDX_SERIES });
    if (!(await goto(page, ev, 'forexfactory%3Aadp%3A2026-08-12'))) { test.skip(true, 'gated'); return; }
    await expect(page.locator('.pub-curve-explain')).toHaveCount(0); // no generic filler
    await expect(page.locator('.pub-var-headline')).toBeVisible();   // variation still shown
    await assertNoForbiddenVocab(page);
    await page.screenshot({ path: `test-results/nw8-nosentence-${vp.tag}.png`, fullPage: true });
  });
}
