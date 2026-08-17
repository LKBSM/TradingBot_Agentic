import { test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * VZ-2 — capture-only spec (no assertions). Renders /zones backend-free with a
 * generously populated reading (many zones, all four groups) so the before/after
 * screenshots show the height/scroll fix and the 2-up grid. Full-page PNGs land
 * in docs/audits/vz-2/. Run BEFORE the CSS change and AFTER it; the file name
 * carries the phase via VZ2_PHASE (before|after).
 */

const PRICE = 2390;
const PHASE = process.env.VZ2_PHASE ?? 'after';

function iso(h: number, m = 0): string {
  return `2026-06-20T${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:00Z`;
}
function ob(over: Record<string, unknown>) {
  return {
    id: 'x', direction: 'bullish', level_high: 0, level_low: 0, importance: 'medium',
    status: 'active', created_at: iso(8), tested: false, user_flagged: false,
    contacts: [], origin: null, ...over,
  };
}
function fvg(over: Record<string, unknown>) {
  return {
    id: 'x', direction: 'bullish', level_high: 0, level_low: 0, status: 'active',
    created_at: iso(8), tested: false, user_flagged: false, contacts: [], ...over,
  };
}

// A busy reading: 2 inside · 7 above · 6 below · 4 consumed = 19 zones.
const order_blocks = [
  ob({
    id: 'ob-inside', level_low: 2388, level_high: 2392, tested: true,
    contacts: [{ at: iso(16, 40), level: 2390, outcome: 'inside' }],
    origin: { kind: 'bos', direction: 'bullish', at: iso(9), level: 2386 },
  }),
  ob({ id: 'ob-inside-2', level_low: 2389, level_high: 2393, direction: 'bearish' }),
  ...Array.from({ length: 4 }, (_, i) =>
    ob({ id: `ob-above-${i}`, level_low: 2400 + i * 12, level_high: 2404 + i * 12, direction: i % 2 ? 'bearish' : 'bullish' }),
  ),
  ...Array.from({ length: 3 }, (_, i) =>
    ob({
      id: `ob-below-${i}`, level_low: 2350 - i * 14, level_high: 2354 - i * 14,
      direction: 'bullish', tested: i === 0,
      contacts: i === 0 ? [{ at: iso(12), level: 2352, outcome: 'edge_touch' }] : [],
    }),
  ),
];
const fair_value_gaps = [
  fvg({
    id: 'fvg-above', level_low: 2408, level_high: 2413, status: 'partially_filled',
    tested: true, fill_level: 2410,
    contacts: [
      { at: iso(10, 30), level: 2410, outcome: 'entry_exit' },
      { at: iso(15), level: 2412.9, outcome: 'edge_touch' },
    ],
  }),
  ...Array.from({ length: 2 }, (_, i) =>
    fvg({ id: `fvg-above-${i}`, level_low: 2455 + i * 10, level_high: 2459 + i * 10, direction: i % 2 ? 'bearish' : 'bullish' }),
  ),
  ...Array.from({ length: 3 }, (_, i) =>
    fvg({ id: `fvg-below-${i}`, level_low: 2320 - i * 12, level_high: 2324 - i * 12, direction: 'bearish' }),
  ),
];
const consumed_fair_value_gaps = Array.from({ length: 4 }, (_, i) =>
  fvg({
    id: `fvg-filled-${i}`, level_low: 2360 - i * 8, level_high: 2364 - i * 8, status: 'filled', tested: true,
    contacts: [
      { at: iso(3), level: 2362 - i * 8, outcome: 'entry_exit' },
      { at: iso(8, 15), level: 2360 - i * 8, outcome: 'traversal' },
    ],
  }),
);

const READING = {
  ...FIXTURE_XAU_M15,
  header: { ...FIXTURE_XAU_M15.header, close_price: PRICE },
  structure: {
    ...FIXTURE_XAU_M15.structure,
    order_blocks,
    fair_value_gaps,
    consumed_order_blocks: [],
    consumed_fair_value_gaps,
    liquidity_pools: [
      { id: 'liq-1', side: 'bsl', kind: 'equal_highs', level: 2406, touches: 2, is_external: true, status: 'intact', created_at: iso(7), user_flagged: false },
    ],
  },
};
const SIBLING = {
  ...READING,
  structure: {
    ...READING.structure,
    order_blocks: [ob({ id: 'ob-h1-wrap', level_low: 2387, level_high: 2393 })],
    fair_value_gaps: [fvg({ id: 'fvg-h1-wrap', level_low: 2406, level_high: 2414 })],
    consumed_order_blocks: [],
    consumed_fair_value_gaps: [],
    liquidity_pools: [],
  },
};

function candles() {
  const start = Math.floor(Date.UTC(2026, 5, 20) / 1000);
  const c = Array.from({ length: 10 }, (_, i) => ({
    time: start + i * 900, open: PRICE, high: PRICE + 1, low: PRICE - 1, close: PRICE, volume: 100,
  }));
  return { instrument: 'XAUUSD', timeframe: 'M15', candles: c };
}

async function mock(page: Page) {
  await page.route('**/api/candles**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(candles()) }),
  );
  await page.route('**/api/market-reading**', (route) => {
    const body = /timeframe=M15(\b|&|$)/.test(route.request().url()) ? READING : SIBLING;
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
  });
}

const VIEWPORTS = [
  { name: '1280x800', width: 1280, height: 800 },
  { name: '390x844', width: 390, height: 844 },
];
const LOCALES = [
  { name: 'fr', path: '/zones?instrument=XAUUSD&timeframe=M15' },
  { name: 'en', path: '/en/zones?instrument=XAUUSD&timeframe=M15' },
];

for (const vp of VIEWPORTS) {
  for (const loc of LOCALES) {
    test(`shot ${PHASE} ${loc.name} ${vp.name}`, async ({ page }) => {
      test.setTimeout(90_000);
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await mock(page);
      await page.goto(loc.path, { waitUntil: 'domcontentloaded' });
      await dismissCookieBanner(page);
      await page.locator('[data-zone-id="ob-inside"]').first().waitFor({ state: 'visible', timeout: 60_000 });
      await page.waitForTimeout(600);
      await page.screenshot({
        path: `../docs/audits/vz-2/${PHASE}-${loc.name}-${vp.name}.png`,
        fullPage: true,
      });
      // Also a viewport-only shot (what the fold actually shows — the 4-card test).
      await page.screenshot({
        path: `../docs/audits/vz-2/${PHASE}-${loc.name}-${vp.name}-fold.png`,
        fullPage: false,
      });
    });
  }
}
