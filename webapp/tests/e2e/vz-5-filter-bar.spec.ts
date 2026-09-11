import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * VZ-5 — the /zones filter bar, backend-free, at the two reference viewports.
 *
 * Cible : docs/design/zones_bar_v2.html. What is asserted here is the SHAPE the
 * mission asked for, not pixel values:
 *   • no uppercase étiquette above any control group;
 *   • filters and timeframes are the same segmented control;
 *   • the sort is a discreet dropdown, not a third row of equal-weight pills;
 *   • the header keeps only the zone count, as plain text, no framed pill;
 *   • the price freshness is a caption UNDER the bar;
 *   • the group heading reads in sentence case.
 *
 * It also re-proves commit A in a real browser: pinning a market from this very
 * dropdown lists it ONCE.
 */

const PRICE = 2390;

function iso(h: number, m = 0): string {
  return `2026-06-20T${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:00Z`;
}

const order_blocks = [
  { id: 'z-a', low: 2388, high: 2392 },
  { id: 'z-b', low: 2385, high: 2389 },
  { id: 'z-c', low: 2391, high: 2395 },
].map((s, i) => ({
  id: s.id,
  direction: i % 2 ? 'bearish' : 'bullish',
  level_low: s.low,
  level_high: s.high,
  importance: 'medium',
  status: 'active',
  created_at: iso(8),
  tested: false,
  user_flagged: false,
  contacts: [],
  origin: null,
}));

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

function candles() {
  const start = Math.floor(Date.UTC(2026, 5, 20) / 1000);
  return {
    instrument: 'XAUUSD',
    timeframe: 'M15',
    candles: Array.from({ length: 10 }, (_, i) => ({
      time: start + i * 900,
      open: PRICE,
      high: PRICE + 1,
      low: PRICE - 1,
      close: PRICE,
      volume: 100,
    })),
  };
}

async function mock(page: Page) {
  await page.route('**/api/candles**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(candles()) }),
  );
  await page.route('**/api/market-reading**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(READING) }),
  );
}

const VIEWPORTS = [
  { name: '1280x800', width: 1280, height: 800 },
  { name: '390x844', width: 390, height: 844 },
];

const PATH = '/zones?instrument=XAUUSD&timeframe=M15';

async function open(page: Page, width: number, height: number) {
  await page.setViewportSize({ width, height });
  await mock(page);
  await page.goto(PATH, { waitUntil: 'domcontentloaded' });
  await dismissCookieBanner(page);
  await page.locator('.zbar').first().waitFor({ state: 'visible', timeout: 90_000 });
}

for (const vp of VIEWPORTS) {
  test(`VZ-5 — bar shape ${vp.name}`, async ({ page }) => {
    test.setTimeout(120_000);
    await open(page, vp.width, vp.height);

    const bar = page.locator('.zbar').first();

    // 1 — the market selector and the filters live on the SAME row, and the
    //     filters are a segmented group named for assistive tech.
    await expect(bar.locator('[data-testid="mkt-selector-bar"]')).toBeVisible();
    await expect(bar.getByRole('group', { name: 'Filtrer les zones' })).toBeVisible();

    // 2 — no uppercase étiquette above any group. The three shouting labels the
    //     mission named ("FILTRE", "TRI", "LE PRIX EST DEDANS") are gone as such.
    await expect(bar.locator('.uppercase')).toHaveCount(0);
    await expect(page.getByText('FILTRE', { exact: true })).toHaveCount(0);
    await expect(page.getByText('TRI', { exact: true })).toHaveCount(0);

    // 3 — the sort is a dropdown, not a row of pills at equal visual weight.
    const sort = page.locator('#zones-sort');
    await expect(sort).toBeVisible();
    expect(await sort.evaluate((el) => el.tagName)).toBe('SELECT');
    await expect(bar.getByRole('button', { name: 'Proximité' })).toHaveCount(0);

    // 4 — the header keeps ONLY the zone count, unframed.
    await expect(page.locator('.pghead .livebadge')).toHaveCount(0);
    const status = page.getByTestId('zones-status');
    await expect(status).toBeVisible();
    await expect(status).toContainText(/zones? suivies?/);
    await expect(status).not.toContainText('XAU/USD');

    // 5 — the freshness line is a caption UNDER the bar, not stacked in the
    //     header column. (It renders only when the reference price carries a
    //     timestamp — assert placement only when it is there.)
    const fresh = page.locator('.zfresh');
    if (await fresh.count()) {
      await expect(page.locator('.pghead .zfresh')).toHaveCount(0);
      const order = await page.evaluate(() => {
        const b = document.querySelector('.zbar');
        const f = document.querySelector('.zfresh');
        if (!b || !f) return 0;
        // 4 === DOCUMENT_POSITION_FOLLOWING → .zfresh comes after .zbar.
        return b.compareDocumentPosition(f) & 4;
      });
      expect(order).toBe(4);
    }

    // 6 — the group heading reads in sentence case: the uppercasing was CSS.
    const sep = page.locator('.zsep').first();
    await expect(sep).toBeVisible();
    expect(await sep.evaluate((el) => getComputedStyle(el).textTransform)).toBe('none');

    await page.screenshot({
      path: `../docs/audits/vz-5/bar-${vp.name}.png`,
      fullPage: false,
    });
  });

  test(`VZ-5 — sorting still works ${vp.name}`, async ({ page }) => {
    test.setTimeout(120_000);
    await open(page, vp.width, vp.height);

    const sort = page.locator('#zones-sort');
    await expect(sort).toHaveValue('proximity');
    await sort.selectOption('formation');
    await expect(sort).toHaveValue('formation');
    // Re-sorting reorders; it never filters anything away.
    await expect(page.locator('article[data-zone-id]')).toHaveCount(order_blocks.length);
  });
}

test('VZ-5 — pinning from the /zones dropdown lists the market ONCE (commit A)', async ({
  page,
}) => {
  test.setTimeout(120_000);
  await open(page, 1280, 800);

  await page.locator('[data-testid="mkt-selector-bar"]').getByRole('button', { name: 'Marchés' }).click();

  const dropdown = page.locator('[data-testid="mkt-selector-bar"]');
  const eur = dropdown.getByText('Euro / Dollar (EUR/USD)', { exact: true });
  await expect(eur).toHaveCount(1);

  await dropdown.getByRole('button', { name: /Épingler Euro \/ Dollar/i }).click();

  // It moved INTO « Épinglés » — it was not added on top of the full list.
  await expect(dropdown.getByText('Épinglés', { exact: true })).toBeVisible();
  await expect(eur).toHaveCount(1);

  await page.screenshot({ path: '../docs/audits/vz-5/pinned-once-1280x800.png' });
});
