import { test, expect, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

// Reuse the busy reading from the shots spec via a tiny inline copy.
const PRICE = 2390;
function iso(h: number, m = 0): string {
  return `2026-06-20T${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:00Z`;
}
function ob(over: Record<string, unknown>) {
  return { id: 'x', direction: 'bullish', level_high: 0, level_low: 0, importance: 'medium', status: 'active', created_at: iso(8), tested: false, user_flagged: false, contacts: [], origin: null, ...over };
}
function fvg(over: Record<string, unknown>) {
  return { id: 'x', direction: 'bullish', level_high: 0, level_low: 0, status: 'active', created_at: iso(8), tested: false, user_flagged: false, contacts: [], ...over };
}
const READING = {
  ...FIXTURE_XAU_M15,
  header: { ...FIXTURE_XAU_M15.header, close_price: PRICE },
  structure: {
    ...FIXTURE_XAU_M15.structure,
    order_blocks: [
      ob({ id: 'ob-inside', level_low: 2388, level_high: 2392, tested: true, contacts: [{ at: iso(16, 40), level: 2390, outcome: 'inside' }], origin: { kind: 'bos', direction: 'bullish', at: iso(9), level: 2386 } }),
      ob({ id: 'ob-inside-2', level_low: 2389, level_high: 2393, direction: 'bearish' }),
      ...Array.from({ length: 4 }, (_, i) => ob({ id: `ob-above-${i}`, level_low: 2400 + i * 12, level_high: 2404 + i * 12, direction: i % 2 ? 'bearish' : 'bullish' })),
      ...Array.from({ length: 3 }, (_, i) => ob({ id: `ob-below-${i}`, level_low: 2350 - i * 14, level_high: 2354 - i * 14, direction: 'bullish', tested: i === 0, contacts: i === 0 ? [{ at: iso(12), level: 2352, outcome: 'edge_touch' }] : [] })),
    ],
    fair_value_gaps: [
      fvg({ id: 'fvg-above', level_low: 2408, level_high: 2413, status: 'partially_filled', tested: true, fill_level: 2410, contacts: [{ at: iso(10, 30), level: 2410, outcome: 'entry_exit' }, { at: iso(15), level: 2412.9, outcome: 'edge_touch' }] }),
      ...Array.from({ length: 2 }, (_, i) => fvg({ id: `fvg-above-${i}`, level_low: 2455 + i * 10, level_high: 2459 + i * 10, direction: i % 2 ? 'bearish' : 'bullish' })),
      ...Array.from({ length: 3 }, (_, i) => fvg({ id: `fvg-below-${i}`, level_low: 2320 - i * 12, level_high: 2324 - i * 12, direction: 'bearish' })),
    ],
    consumed_order_blocks: [],
    consumed_fair_value_gaps: Array.from({ length: 4 }, (_, i) => fvg({ id: `fvg-filled-${i}`, level_low: 2360 - i * 8, level_high: 2364 - i * 8, status: 'filled', tested: true, contacts: [{ at: iso(3), level: 2362 - i * 8, outcome: 'entry_exit' }, { at: iso(8, 15), level: 2360 - i * 8, outcome: 'traversal' }] })),
    liquidity_pools: [{ id: 'liq-1', side: 'bsl', kind: 'equal_highs', level: 2406, touches: 2, is_external: true, status: 'intact', created_at: iso(7), user_flagged: false }],
  },
};
const SIBLING = { ...READING, structure: { ...READING.structure, order_blocks: [ob({ id: 'ob-h1-wrap', level_low: 2387, level_high: 2393 })], fair_value_gaps: [fvg({ id: 'fvg-h1-wrap', level_low: 2406, level_high: 2414 })], consumed_order_blocks: [], consumed_fair_value_gaps: [], liquidity_pools: [] } };
function candles() {
  const start = Math.floor(Date.UTC(2026, 5, 20) / 1000);
  return { instrument: 'XAUUSD', timeframe: 'M15', candles: Array.from({ length: 10 }, (_, i) => ({ time: start + i * 900, open: PRICE, high: PRICE + 1, low: PRICE - 1, close: PRICE, volume: 100 })) };
}
async function mock(page: Page) {
  await page.route('**/api/candles**', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(candles()) }));
  await page.route('**/api/market-reading**', (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(/timeframe=M15(\b|&|$)/.test(r.request().url()) ? READING : SIBLING) }));
}

test('measure visible cards + single scroll container @1280x800', async ({ page }) => {
  test.setTimeout(90_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  await mock(page);
  await page.goto('/zones?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
  await dismissCookieBanner(page);
  await page.locator('[data-zone-id="ob-inside"]').first().waitFor({ state: 'visible', timeout: 60_000 });
  await page.waitForTimeout(500);

  const data = await page.evaluate(() => {
    const H = 800;
    const cards = Array.from(document.querySelectorAll('.zone')) as HTMLElement[];
    const rects = cards.map((c) => c.getBoundingClientRect());
    const fullyVisible = rects.filter((r) => r.top >= 0 && r.bottom <= H).length;
    // Every element that OWNS a scrollbar inside the shell — the list must add
    // none of its own; only the centre column scrolls (VZ-2 single scroll).
    const scrollers = (Array.from(document.querySelectorAll('.app-shell *')) as HTMLElement[])
      .filter((el) => {
        const oy = getComputedStyle(el).overflowY;
        return (oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight + 2;
      })
      .map((el) => (typeof el.className === 'string' ? el.className : ''));
    // The cards column and its sub-parts must NOT be scroll containers.
    const listScrolls = ['.zcol', '.zcards', '.zgroup'].some((sel) => {
      const el = document.querySelector(sel) as HTMLElement | null;
      if (!el) return false;
      const oy = getComputedStyle(el).overflowY;
      return oy === 'auto' || oy === 'scroll';
    });
    const zcardsEl = document.querySelector('.zcards') as HTMLElement | null;
    const columns = zcardsEl ? getComputedStyle(zcardsEl).gridTemplateColumns : '';
    return { total: cards.length, fullyVisible, scrollers, listScrolls, columns };
  });

  // The mission's hard targets.
  expect(data.fullyVisible).toBeGreaterThanOrEqual(4);
  // Single scroll container: the list area introduces no inner scroll of its own.
  expect(data.listScrolls).toBe(false);
  // 2-up above the threshold: two grid tracks at 1280px.
  expect(data.columns.trim().split(/\s+/)).toHaveLength(2);
});
