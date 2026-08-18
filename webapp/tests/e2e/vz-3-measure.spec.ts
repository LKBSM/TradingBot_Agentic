import { test, expect, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * VZ-3 — assertions on the rebuilt gauge (after only). One card per state; we
 * check the bracket is present exactly where a distance exists, and that at
 * 390 px no two gauge labels overlap (the mission's « aucune étiquette ne se
 * superpose » rule).
 */

const PRICE = 2390;
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
  { id: 'z-inside', low: 2388, high: 2392, bracket: false },
  { id: 'z-above', low: 2385, high: 2389, bracket: true },
  { id: 'z-below', low: 2391, high: 2395, bracket: true },
  { id: 'z-out-above', low: 2360, high: 2364, bracket: false },
  { id: 'z-out-below', low: 2420, high: 2424, bracket: false },
];
const order_blocks = STATES.map((s, i) =>
  ob({
    id: s.id, level_low: s.low, level_high: s.high, direction: i % 2 ? 'bearish' : 'bullish',
    tested: s.id === 'z-inside',
    contacts: s.id === 'z-inside' ? [{ at: iso(16, 40), level: PRICE, outcome: 'inside' }] : [],
  }),
);
const READING = {
  ...FIXTURE_XAU_M15,
  header: { ...FIXTURE_XAU_M15.header, close_price: PRICE },
  structure: {
    ...FIXTURE_XAU_M15.structure,
    order_blocks, fair_value_gaps: [], consumed_order_blocks: [], consumed_fair_value_gaps: [], liquidity_pools: [],
  },
};
const SIBLING = { ...READING, structure: { ...READING.structure, order_blocks: [], fair_value_gaps: [] } };
function candles() {
  const start = Math.floor(Date.UTC(2026, 5, 20) / 1000);
  return { instrument: 'XAUUSD', timeframe: 'M15', candles: Array.from({ length: 10 }, (_, i) => ({ time: start + i * 900, open: PRICE, high: PRICE + 1, low: PRICE - 1, close: PRICE, volume: 100 })) };
}
async function mock(page: Page) {
  await page.route('**/api/candles**', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(candles()) }));
  await page.route('**/api/market-reading**', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(/timeframe=M15(\b|&|$)/.test(r.request().url()) ? READING : SIBLING) }));
}

test('bracket appears only where a distance exists @1280x800', async ({ page }) => {
  test.setTimeout(120_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  await mock(page);
  await page.goto('/zones?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
  await dismissCookieBanner(page);
  await page.locator('[data-zone-id="z-inside"]').first().waitFor({ state: 'visible', timeout: 90_000 });
  for (const s of STATES) {
    const card = page.locator(`[data-zone-id="${s.id}"]`).first();
    const brk = card.locator('.gbrk');
    if (s.bracket) {
      await expect(brk, `${s.id} should have a bracket`).toHaveCount(1);
    } else {
      await expect(brk, `${s.id} should NOT have a bracket`).toHaveCount(0);
    }
    // The gauge itself is always present (price is known for every card).
    await expect(card.locator('.zgauge')).toHaveCount(1);
  }
});

test('no gauge label overlaps another @390x844', async ({ page }) => {
  test.setTimeout(120_000);
  await page.setViewportSize({ width: 390, height: 844 });
  await mock(page);
  await page.goto('/zones?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
  await dismissCookieBanner(page);
  await page.locator('[data-zone-id="z-inside"]').first().waitFor({ state: 'visible', timeout: 90_000 });
  await page.waitForTimeout(400);

  const overlaps = await page.evaluate(() => {
    const bad: string[] = [];
    const gauges = Array.from(document.querySelectorAll('.zgauge')) as HTMLElement[];
    const intersects = (a: DOMRect, b: DOMRect) =>
      a.left < b.right - 0.5 && b.left < a.right - 0.5 && a.top < b.bottom - 0.5 && b.top < a.bottom - 0.5;
    for (const g of gauges) {
      const labels = Array.from(
        g.querySelectorAll('.gpx-lbl, .gbrk-lbl, .gedge, .gpx-out'),
      ) as HTMLElement[];
      const rects = labels.map((l) => l.getBoundingClientRect());
      for (let i = 0; i < rects.length; i++) {
        for (let j = i + 1; j < rects.length; j++) {
          const a = rects[i];
          const b = rects[j];
          if (a && b && intersects(a, b)) {
            bad.push(`${g.getAttribute('aria-label')?.slice(0, 24)} :: ${labels[i]!.className} × ${labels[j]!.className}`);
          }
        }
      }
    }
    return bad;
  });
  expect(overlaps, `overlapping labels: ${overlaps.join(' | ')}`).toEqual([]);
});
