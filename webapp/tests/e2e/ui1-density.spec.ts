import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15, FIXTURE_QUIET_XAU_H4 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * UI-1 — Densité et lisibilité de /zones et /scanner. Présentation uniquement.
 *
 * The assertions are LANGUAGE-AGNOSTIC on purpose (computed font sizes, card
 * counts, overflow, absence of raw i18n keys, block presence) so the same suite
 * guards both locales without pinning translated copy. What it locks in:
 *   · no horizontal overflow / no truncation at 1280×800, 1440×900, 390×844;
 *   · the scanner block labels are small étiquettes (≤10px) — they used to fall
 *     back to the browser default 16px and dominate the card;
 *   · on a zone card the price band is the largest text (hierarchy from contrast);
 *   · two scanner results fit at 1280×800; a zone card stays compact (≤360px);
 *   · the three scanner blocks (correspond / à l'encontre / contexte) all render;
 *   · on mobile the rail collapses (no 232px column crushing the content).
 * Live data is validated with the founder before merge (repo e2e convention).
 */

const CONFIG_KEY = 'mia.conditionsConfig.v1';
const CONFIG = {
  logic: 'AND',
  conditions: [{ type: 'trend_is', trend: 'bearish' }, { type: 'price_in_ob' }],
};

function makeCandles(n = 150) {
  const start = Math.floor(Date.UTC(2026, 5, 20) / 1000);
  return {
    instrument: 'XAUUSD',
    timeframe: 'M15',
    candles: Array.from({ length: n }, (_, i) => {
      const close = 2300 + i * 2;
      return {
        time: start + i * 900,
        open: close - 0.5,
        high: close + 1,
        low: close - 1,
        close,
        volume: 100,
      };
    }),
  };
}

async function mockZones(page: Page) {
  await page.route('**/api/access/me', (r) =>
    r.fulfill({
      json: { has_full_access: true, entitlements: { instruments: [], timeframes: [] } },
    }),
  );
  await page.route('**/api/candles**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(makeCandles()),
    }),
  );
  await page.route('**/api/market-reading**', (r) => {
    const full = /timeframe=M15/.test(r.request().url());
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(full ? FIXTURE_XAU_M15 : FIXTURE_QUIET_XAU_H4),
    });
  });
}

const CTX = {
  trend: 'bearish',
  market_phase: 'trend',
  volatility_observed: 'normal',
  mtf_confluence: {},
  mtf_trends: { h4: 'bearish', h1: 'bearish', m15: 'bearish' },
  bos: { direction: 'bearish' },
  choch: null,
  active_order_blocks: 2,
  active_fair_value_gaps: 1,
  structural_range: { low: 4010, high: 4055 },
  news_upcoming: [{ impact: 'high' }],
};
function match(over: Record<string, unknown> = {}) {
  return {
    instrument: 'XAUUSD',
    timeframe: 'M15',
    candle_close_ts: new Date().toISOString(),
    close_price: 4029,
    matched: true,
    met_count: 2,
    total: 2,
    non_evaluable_count: 0,
    conditions_met: [
      { type: 'trend_is', label: 'La tendance', met: true, detail: 'Baissière.' },
      {
        type: 'price_in_ob',
        label: 'Prix dans un OB',
        met: true,
        detail: 'OB 4028–4030.',
      },
    ],
    conditions_unmet: [],
    conditions_non_evaluable: [],
    context_against: [
      { label: 'Le 4 h est en tendance haussière', detail: 'désaccord multi-unités' },
    ],
    context: CTX,
    freshness: 'fresh',
    bars_behind: 0,
    ...over,
  };
}
const RESULTS = {
  as_of: new Date().toISOString(),
  logic: 'AND',
  scanned: 3,
  matches: [
    match(),
    match({
      instrument: 'EURUSD',
      timeframe: 'H4',
      matched: false,
      met_count: 1,
      total: 2,
      conditions_met: [
        { type: 'trend_is', label: 'La tendance', met: true, detail: 'Baissière.' },
      ],
      conditions_unmet: [
        { type: 'price_in_ob', label: 'Prix dans un OB', met: false, detail: 'Hors OB.' },
      ],
    }),
    match({ instrument: 'XAUUSD', timeframe: 'H1' }),
  ],
  unavailable: [],
};
const NO_COMBO = {
  as_of: new Date().toISOString(),
  logic: 'AND',
  scanned: 1,
  matches: [
    match({
      matched: false,
      met_count: 1,
      total: 2,
      conditions_met: [
        { type: 'trend_is', label: 'La tendance', met: true, detail: 'Baissière.' },
      ],
      conditions_unmet: [
        { type: 'price_in_ob', label: 'Prix dans un OB', met: false, detail: 'Hors OB.' },
      ],
    }),
  ],
  unavailable: [],
};

async function mockScanner(page: Page, scan: unknown) {
  await page.route('**/api/access/me', (r) =>
    r.fulfill({
      json: { has_full_access: true, entitlements: { instruments: [], timeframes: [] } },
    }),
  );
  await page.route('**/api/conditions-scan', (r) => r.fulfill({ json: scan }));
  await page.addInitScript(
    (cfg) => window.localStorage.setItem('mia.conditionsConfig.v1', JSON.stringify(cfg)),
    CONFIG,
  );
}

const RAW_KEY_SCAN = `(() => {
  const bad = []; const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); let n;
  while ((n = w.nextNode())) {
    const t = (n.textContent || '').trim();
    if (!t || t.includes('/') || t.includes('@') || t.includes(':')) continue;
    if (/^[a-z][a-zA-Z0-9]*(\\.[a-zA-Z][a-zA-Z0-9]*)+$/.test(t)) bad.push(t);
  }
  return bad;
})()`;

const overflow = (page: Page) =>
  page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
const fontPx = (page: Page, sel: string) =>
  page.evaluate((s) => {
    const el = document.querySelector(s);
    return el ? parseFloat(getComputedStyle(el).fontSize) : null;
  }, sel);

const VIEWPORTS = [
  { name: '1280×800', width: 1280, height: 800, desktop: true },
  { name: '1440×900', width: 1440, height: 900, desktop: true },
  { name: '390×844', width: 390, height: 844, desktop: false },
];
const LOCALES = [
  { code: 'fr', prefix: '' },
  { code: 'en', prefix: '/en' },
];

for (const vp of VIEWPORTS) {
  for (const loc of LOCALES) {
    test.describe(`${vp.name} · ${loc.code}`, () => {
      test.beforeEach(async ({ page }) =>
        page.setViewportSize({ width: vp.width, height: vp.height }),
      );

      test('zones: dense card, price is the largest text, no overflow/keys', async ({
        page,
      }) => {
        test.setTimeout(90_000);
        await mockZones(page);
        await page.goto(`${loc.prefix}/zones?instrument=XAUUSD&timeframe=M15`, {
          waitUntil: 'domcontentloaded',
        });
        await dismissCookieBanner(page);
        await page
          .locator('.zone')
          .first()
          .waitFor({ state: 'visible', timeout: 60_000 });

        expect(await overflow(page)).toBeLessThanOrEqual(1);
        expect(await page.evaluate(RAW_KEY_SCAN)).toEqual([]);

        // Hierarchy from contrast: the price band outweighs the block étiquettes.
        const price = await fontPx(page, '.zone .rng');
        const label = await fontPx(page, '.zone .zpxr .k');
        expect(price).not.toBeNull();
        expect(label).not.toBeNull();
        expect(price!).toBeGreaterThan(label!);
        expect(label!).toBeLessThanOrEqual(10); // étiquette, not a title

        if (vp.desktop) {
          expect(price!).toBe(13);
          // Density: a full zone card stays compact (was ~386px pre-UI-1).
          const h = await page.evaluate(() =>
            Math.round(document.querySelector('.zone')!.getBoundingClientRect().height),
          );
          expect(h).toBeLessThanOrEqual(360);
        }
      });

      test('scanner: 3 blocks render, labels are étiquettes, 2 results fit, no overflow/keys', async ({
        page,
      }) => {
        test.setTimeout(90_000);
        await mockScanner(page, RESULTS);
        await page.goto(`${loc.prefix}/scanner`, { waitUntil: 'domcontentloaded' });
        await dismissCookieBanner(page);
        await page
          .locator('.combo')
          .first()
          .waitFor({ state: 'visible', timeout: 60_000 });

        expect(await overflow(page)).toBeLessThanOrEqual(1);
        expect(await page.evaluate(RAW_KEY_SCAN)).toEqual([]);

        // The three blocks all render, and the « à l'encontre » one is present
        // (never hidden/collapsible).
        const firstCard = page.locator('.combo').first();
        expect(await firstCard.locator('.blk-lbl').count()).toBeGreaterThanOrEqual(3);
        await expect(page.getByTestId('against-block').first()).toBeVisible();

        // The block label bug: it fell back to 16px. It must now be a small étiquette.
        const blk = await fontPx(page, '.combo .blk-lbl');
        const nm = await fontPx(page, '.combo .nm');
        expect(blk).not.toBeNull();
        expect(blk!).toBeLessThanOrEqual(10);
        expect(nm!).toBeGreaterThan(blk!); // market name stays primary

        if (vp.desktop) {
          const visible = await page.evaluate(() => {
            const vh = window.innerHeight;
            return Array.from(document.querySelectorAll('.combo')).filter((c) => {
              const r = c.getBoundingClientRect();
              return r.top >= 0 && r.bottom <= vh + 0.5;
            }).length;
          });
          expect(visible).toBeGreaterThanOrEqual(2);
        }
      });
    });
  }
}

// ── State coverage: expanded card, consumed group, empty scanner, mobile rail ──

test.describe('states', () => {
  test('zones: expanding a card reveals the details grid (fold stays a fold)', async ({
    page,
  }) => {
    test.setTimeout(90_000);
    await page.setViewportSize({ width: 1280, height: 800 });
    await mockZones(page);
    await page.goto('/zones?instrument=XAUUSD&timeframe=M15', {
      waitUntil: 'domcontentloaded',
    });
    await dismissCookieBanner(page);
    const card = page.locator('.zone').first();
    await card.waitFor({ state: 'visible', timeout: 60_000 });
    await expect(card.locator('.zkv')).toHaveCount(0); // collapsed by default
    await card.locator('.zdeth').click();
    await expect(card.locator('.zkv')).toBeVisible(); // expanded on demand
  });

  test('scanner: no-combo is an explicit non-error state', async ({ page }) => {
    test.setTimeout(90_000);
    await page.setViewportSize({ width: 1280, height: 800 });
    await mockScanner(page, NO_COMBO);
    await page.goto('/scanner', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);
    await expect(page.getByTestId('scan-no-combo')).toBeVisible();
    expect(await overflow(page)).toBeLessThanOrEqual(1);
  });

  test('mobile: the rail collapses (no 232px column) on /zones and /scanner', async ({
    page,
  }) => {
    test.setTimeout(90_000);
    await page.setViewportSize({ width: 390, height: 844 });
    await mockZones(page);
    await page.goto('/zones?instrument=XAUUSD&timeframe=M15', {
      waitUntil: 'domcontentloaded',
    });
    await dismissCookieBanner(page);
    await page.locator('.zone').first().waitFor({ state: 'visible', timeout: 60_000 });
    await expect(page.locator('.app-shell .rail')).toBeHidden();
    expect(await overflow(page)).toBeLessThanOrEqual(1);

    await mockScanner(page, RESULTS);
    await page.goto('/scanner', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);
    await page.locator('.combo').first().waitFor({ state: 'visible', timeout: 60_000 });
    await expect(page.locator('.app-shell .rail')).toBeHidden();
    expect(await overflow(page)).toBeLessThanOrEqual(1);
  });
});
