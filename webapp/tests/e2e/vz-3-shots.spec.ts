import { test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * VZ-3 — capture-only spec (no assertions). Renders /zones backend-free with a
 * reading crafted so ONE card lands in each gauge state, then screenshots each
 * card on its own. Run BEFORE the change and AFTER it; VZ3_PHASE (before|after)
 * names the files under docs/audits/vz-3/.
 *
 * The single price is 2390. Each zone's geometry is chosen to hit one state:
 *   inside     · price within the band
 *   above      · price above the zone, still in the window
 *   below      · price below the zone, still in the window
 *   out-above  · price far above the zone (out of window)
 *   out-below  · price far below the zone (out of window)
 */

const PRICE = 2390;
const PHASE = process.env.VZ3_PHASE ?? 'after';

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

const STATES = [
  { id: 'z-inside', low: 2388, high: 2392 }, // price 2390 inside
  { id: 'z-above', low: 2385, high: 2389 }, // price above zone, in window (win 2383..2391)
  { id: 'z-below', low: 2391, high: 2395 }, // price below zone, in window (win 2389..2397)
  { id: 'z-out-above', low: 2360, high: 2364 }, // price far above (win 2358..2366)
  { id: 'z-out-below', low: 2420, high: 2424 }, // price far below (win 2418..2426)
];

const order_blocks = STATES.map((s, i) =>
  ob({
    id: s.id, level_low: s.low, level_high: s.high,
    direction: i % 2 ? 'bearish' : 'bullish',
    tested: s.id === 'z-inside',
    contacts: s.id === 'z-inside' ? [{ at: iso(16, 40), level: PRICE, outcome: 'inside' }] : [],
  }),
);

const READING = {
  ...FIXTURE_XAU_M15,
  header: { ...FIXTURE_XAU_M15.header, close_price: PRICE },
  structure: {
    ...FIXTURE_XAU_M15.structure,
    order_blocks,
    fair_value_gaps: [],
    consumed_order_blocks: [],
    consumed_fair_value_gaps: [],
    liquidity_pools: [],
  },
};
const SIBLING = {
  ...READING,
  structure: { ...READING.structure, order_blocks: [], fair_value_gaps: [] },
};

function candles() {
  const start = Math.floor(Date.UTC(2026, 5, 20) / 1000);
  return {
    instrument: 'XAUUSD', timeframe: 'M15',
    candles: Array.from({ length: 10 }, (_, i) => ({
      time: start + i * 900, open: PRICE, high: PRICE + 1, low: PRICE - 1, close: PRICE, volume: 100,
    })),
  };
}

async function mock(page: Page) {
  await page.route('**/api/candles**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(candles()) }),
  );
  await page.route('**/api/market-reading**', (r) =>
    r.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify(/timeframe=M15(\b|&|$)/.test(r.request().url()) ? READING : SIBLING),
    }),
  );
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
    test(`gauge ${PHASE} ${loc.name} ${vp.name}`, async ({ page }) => {
      test.setTimeout(120_000);
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await mock(page);
      await page.goto(loc.path, { waitUntil: 'domcontentloaded' });
      await dismissCookieBanner(page);
      await page.locator('[data-zone-id="z-inside"]').first().waitFor({ state: 'visible', timeout: 90_000 });
      await page.waitForTimeout(500);
      // A full-page shot for context…
      await page.screenshot({ path: `../docs/audits/vz-3/${PHASE}-${loc.name}-${vp.name}-full.png`, fullPage: true });
      // …then one tight shot per state card.
      for (const s of STATES) {
        const card = page.locator(`[data-zone-id="${s.id}"]`).first();
        if (await card.count()) {
          await card.scrollIntoViewIfNeeded();
          await card.screenshot({ path: `../docs/audits/vz-3/${PHASE}-${loc.name}-${vp.name}-${s.id}.png` });
        }
      }
    });
  }
}
