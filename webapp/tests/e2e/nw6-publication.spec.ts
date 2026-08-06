import { expect, test, type Page } from '@playwright/test';

/**
 * NW-6 — page de publication : les correctifs de la mission, chacun au 1280×800
 * et au 390×844, avec capture pour la comparaison visuelle à
 * docs/design/reference-publication.html :
 *   A. PASSÉE       — le libellé du compte à rebours lit « Publiée » (Défaut A) ;
 *   B. À VENIR      — le libellé lit « Publication dans » (Défaut A) ;
 *   C. AVEC VALEURS — la courbe est rendue, le point à venir n'a pas de valeur ;
 *   D. SANS VALEURS — le bloc courbe est absent (jamais vide/approximé) ;
 *   E. SANS FICHE   — aucun bloc pédagogique générique (Défaut B).
 * Chaque page ne porte QU'UN bloc d'avertissement, la mention M.I.A conservée
 * (Défaut C). Tout le réseau est mocké ; le vrai backend n'est jamais appelé.
 */

const D = 86400_000;
const iso = (ms: number) => new Date(Date.now() + ms).toISOString();

const FULL_ACCESS = {
  authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false,
  tier: 'owner', is_owner: true, has_full_access: true,
  entitlements: { instruments: null, timeframes: null, scanner: true, chat: { limit: null, used: 0, remaining: null } },
};

const SERIES = [
  { period: '2025-09', value: 2.9 }, { period: '2025-10', value: 3.0 },
  { period: '2025-11', value: 2.8 }, { period: '2025-12', value: 2.9 },
  { period: '2026-01', value: 3.0 }, { period: '2026-02', value: 3.3 },
  { period: '2026-03', value: 3.2 }, { period: '2026-04', value: 3.0 },
  { period: '2026-05', value: 2.9 }, { period: '2026-06', value: 3.0 },
  { period: '2026-07', value: 3.1 },
];

type EventOpts = {
  eventId: string; eventKey: string; source: string; organism: string | null;
  seriesCode: string | null; scheduledMs: number; state: string;
  series: typeof SERIES | [];
};

function makeEvent(o: EventOpts) {
  return {
    window_start: iso(-40 * D), window_end: iso(40 * D), generated_at: iso(0),
    coverage: { source: 'official', feed_start: null, feed_end: null, partial: false, last_success: {}, stale_sources: [] },
    attribution: o.organism ? [{ source: o.source, organism: o.organism, license_label: 'Domaine public (17 U.S.C. §105)', policy_url: `https://www.${o.source}.gov/` }] : [],
    events: [{
      event_id: o.eventId, source: o.source, series_code: o.seriesCode, license_label: 'x',
      event: o.eventKey.toUpperCase(), currency: 'USD', organism: o.organism, periodicity: 'monthly',
      scheduled_at: iso(o.scheduledMs), source_timezone: 'America/New_York', time_confirmed: true,
      markets: ['XAUUSD', 'EURUSD'], value_unit: '% de variation annuelle',
      actual: o.state === 'published' ? 3.1 : null, actual_initial: 3.2, previous: 3.0,
      revised: o.state === 'published', revised_at: iso(-20 * D), actual_state: o.state, refreshed_at: iso(0),
      value_series: o.series,
    }],
  };
}

const EMPTY_MEAS = { event_key: 'x', market: '', calm_before: null, structure_state: null, return_to_calm: null };
const json = (b: unknown) => ({ status: 200, contentType: 'application/json', body: JSON.stringify(b) });

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

async function overflow(page: Page): Promise<number> {
  return page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
}

/** Invariants that must hold on EVERY publication page (Défaut C + honesty). */
async function assertCommonInvariants(page: Page) {
  // Défaut C — exactly ONE page-level warning block, and the M.I.A mention stays.
  expect(await page.locator('.cal-nono').count()).toBe(1);
  await expect(page.locator('.pub-mia-cap')).toBeVisible();
  // No candle / central-statistic wording, no raw i18n key.
  const txt = ((await page.locator('.cald').textContent()) ?? '').toLowerCase();
  for (const w of ['bougie', 'médiane', 'moyenne']) expect(txt).not.toContain(w);
  expect(/\bcalendar\.[a-zA-Z.]+\b/.test(txt)).toBe(false);
}

const CPI = (scheduledMs: number, state: string, series: typeof SERIES | []): EventOpts => ({
  eventId: 'bls:us_cpi:2026-08-12', eventKey: 'us_cpi', source: 'bls',
  organism: 'Bureau of Labor Statistics', seriesCode: 'CUUR0000SA0',
  scheduledMs, state, series,
});
const CPI_URL = 'bls%3Aus_cpi%3A2026-08-12';

for (const vp of [{ w: 1280, h: 800, tag: '1280' }, { w: 390, h: 844, tag: '390' }]) {
  test(`${vp.tag}: A — PAST release reads « Publiée »`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    if (!(await goto(page, makeEvent(CPI(-1 * D, 'published', SERIES)), CPI_URL))) { test.skip(true, 'gated'); return; }
    expect((await page.locator('.cald-cd .k').textContent())?.trim()).toBe('Publiée');
    await assertCommonInvariants(page);
    expect(await overflow(page)).toBeLessThanOrEqual(1);
    await page.screenshot({ path: `test-results/nw6-past-${vp.tag}.png`, fullPage: true });
  });

  test(`${vp.tag}: B — UPCOMING release reads « Publication dans »`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    if (!(await goto(page, makeEvent(CPI(12 * D, 'pending', SERIES)), CPI_URL))) { test.skip(true, 'gated'); return; }
    expect((await page.locator('.cald-cd .k').textContent())?.trim()).toBe('Publication dans');
    await assertCommonInvariants(page);
    expect(await overflow(page)).toBeLessThanOrEqual(1);
    await page.screenshot({ path: `test-results/nw6-upcoming-${vp.tag}.png`, fullPage: true });
  });

  test(`${vp.tag}: C — WITH historical values renders the curve, upcoming point blank`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    if (!(await goto(page, makeEvent(CPI(12 * D, 'pending', SERIES)), CPI_URL))) { test.skip(true, 'gated'); return; }
    await expect(page.locator('.pub-curve-svg')).toBeVisible();
    expect((await page.locator('.pt-upcoming-label').textContent()) ?? '').not.toMatch(/\d/);
    await assertCommonInvariants(page);
    await page.screenshot({ path: `test-results/nw6-withvalues-${vp.tag}.png`, fullPage: true });
  });

  test(`${vp.tag}: D — WITHOUT values omits the curve block entirely`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    if (!(await goto(page, makeEvent(CPI(12 * D, 'pending', [])), CPI_URL))) { test.skip(true, 'gated'); return; }
    await expect(page.locator('.pub-curve-svg')).toHaveCount(0);
    // The pedagogy fiche still renders (us_cpi has one), and the page stays honest.
    await expect(page.locator('.pub-ped-body')).toBeVisible();
    await assertCommonInvariants(page);
    await page.screenshot({ path: `test-results/nw6-novalues-${vp.tag}.png`, fullPage: true });
  });

  test(`${vp.tag}: E — publication WITHOUT a written fiche renders no pedagogy block`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    const ev: EventOpts = {
      eventId: 'forexfactory:adp:2026-08-01', eventKey: 'adp', source: 'forexfactory',
      organism: null, seriesCode: null, scheduledMs: 5 * D, state: 'pending', series: [],
    };
    if (!(await goto(page, makeEvent(ev), 'forexfactory%3Aadp%3A2026-08-01'))) { test.skip(true, 'gated'); return; }
    // Défaut B — no fiche exists → the pedagogy card is not rendered (no filler).
    await expect(page.locator('.pub-ped-body')).toHaveCount(0);
    await assertCommonInvariants(page);
    expect(await overflow(page)).toBeLessThanOrEqual(1);
    await page.screenshot({ path: `test-results/nw6-nofiche-${vp.tag}.png`, fullPage: true });
  });
}
