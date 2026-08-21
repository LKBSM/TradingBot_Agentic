import { test, expect, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * LIVE VERIFICATION (founder request) — proves the EXISTING behaviour, no code
 * change: clicking a structure EVENT (CHOCH/BOS row of the Structure panel)
 * brings the chart into view and frames the camera on that event's confirmation
 * bar WITH context (candles before AND after) + an accent marker + the broken
 * level line. Captures before/after so the reframe is visible.
 *
 * Candles are generated to SPAN the fixture's real event timestamps
 * (current_choch 2026-05-26 09:30 / current_bos 11:15) so the framing lands on
 * loaded data — the vz-1-focus fixture window (2026-06-20) does NOT overlap them
 * and only asserts DOM selection, not the visual frame.
 */

// Repo-root docs/audits (Playwright cwd is webapp/) — matches the project's
// audit-shots convention (docs/audits/*-shots), not a webapp-local docs tree.
const SHOTS = '../docs/audits/structure-events-shots';

/** M15 candles 2026-05-26 00:00 → 12:00 UTC (49 bars), price 2373 → 2393 so the
 *  event levels (choch 2384.2 · bos 2391.5) are crossed and sit inside range. */
function makeCandles() {
  const start = Math.floor(Date.UTC(2026, 4, 26, 0, 0) / 1000);
  const n = 49;
  const candles = Array.from({ length: n }, (_, i) => {
    const close = 2373 + (2393 - 2373) * (i / (n - 1));
    const open = i === 0 ? close : 2373 + (2393 - 2373) * ((i - 1) / (n - 1));
    const high = Math.max(open, close) + 0.8;
    const low = Math.min(open, close) - 0.8;
    return { time: start + i * 900, open, high, low, close, volume: 100 };
  });
  return { instrument: 'XAUUSD', timeframe: 'M15', candles };
}

async function mockReading(page: Page) {
  await page.route('**/api/candles**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(makeCandles()) }),
  );
  await page.route('**/api/market-reading**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FIXTURE_XAU_M15) }),
  );
  // Keep the market "open" so the live badge path renders normally (best effort;
  // absent mock just falls back, like vz-1-focus).
  await page.route('**/api/market-status**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ is_open: true, state: 'open', next_open_ts: null, next_close_ts: null }),
    }),
  );
}

test.describe('Structure events — click frames the chart (1280×800)', () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test('CHOCH then BOS row click → chart reframes with context', async ({ page }) => {
    test.setTimeout(90_000);
    await mockReading(page);
    await page.goto('/app?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);

    const chart = page.getByRole('application').first();
    await chart.waitFor({ state: 'visible', timeout: 60_000 });
    // Let the candles settle before the baseline capture.
    await page.waitForTimeout(1500);
    await chart.screenshot({ path: `${SHOTS}/01-desktop-initial.png` });

    const choch = page.locator('.strow').filter({ hasText: 'CHOCH' }).first();
    await choch.scrollIntoViewIfNeeded();
    await choch.click();
    await expect(choch).toHaveAttribute('aria-pressed', 'true');
    await page.waitForTimeout(900); // camera tween ~400ms + settle
    await chart.screenshot({ path: `${SHOTS}/02-desktop-choch-framed.png` });

    const bos = page.locator('.strow').filter({ hasText: 'BOS' }).first();
    await bos.click();
    await expect(bos).toHaveAttribute('aria-pressed', 'true');
    await expect(choch).toHaveAttribute('aria-pressed', 'false'); // single selection
    await page.waitForTimeout(900);
    await chart.screenshot({ path: `${SHOTS}/03-desktop-bos-framed.png` });

    // Re-click deselects → chart restores the pre-selection view.
    await bos.click();
    await expect(bos).toHaveAttribute('aria-pressed', 'false');
    await page.waitForTimeout(900);
    await chart.screenshot({ path: `${SHOTS}/04-desktop-restored.png` });
  });
});

/**
 * OLD event OUTSIDE the loaded candle window: clicking it must page history
 * backward (loadOlder → `?before=`) until its bar is loaded, THEN frame it —
 * never land on empty space to the left of the data. We assert the paging is
 * triggered BY the click (not on initial load) and capture the reframe.
 */
test.describe('Structure events — old event out of window pages history (1280×800)', () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  const CLOSE = Math.floor(Date.UTC(2026, 4, 26, 12, 0) / 1000);
  const RECENT_N = 150;
  const RECENT_START = CLOSE - (RECENT_N - 1) * 900;
  const OLDER_N = 200;
  const OLDER_START = RECENT_START - OLDER_N * 900;
  // Event bar sits INSIDE the older page, well before the recent window's first bar.
  const EVENT_SEC = RECENT_START - 100 * 900;
  const priceAt = (t: number) => 2360 + ((t - OLDER_START) / 900) * 0.15;
  const bar = (t: number) => {
    const close = priceAt(t);
    const open = priceAt(t - 900);
    return { time: t, open, high: Math.max(open, close) + 0.6, low: Math.min(open, close) - 0.6, close, volume: 100 };
  };
  const window = (start: number, n: number) =>
    Array.from({ length: n }, (_, i) => bar(start + i * 900));

  const readingWithOldChoch = () => ({
    ...FIXTURE_XAU_M15,
    header: { ...FIXTURE_XAU_M15.header, candle_close_ts: '2026-05-26T12:00:00+00:00' },
    structure: {
      ...FIXTURE_XAU_M15.structure,
      current_bos: null,
      current_choch: null,
      bos_events: [],
      // A single, OLD CHOCH whose bar is older than the recent candle window.
      choch_events: [
        {
          direction: 'bearish' as const,
          level: Math.round(priceAt(EVENT_SEC) * 100) / 100,
          broken_at: new Date(EVENT_SEC * 1000).toISOString(),
          validation_status: 'confirmed' as const,
        },
      ],
    },
  });

  test('clicking an old CHOCH loads older candles then frames it', async ({ page }) => {
    test.setTimeout(90_000);
    let beforeRequested = false;
    await page.route('**/api/candles**', (route) => {
      const url = route.request().url();
      if (url.includes('before=')) {
        beforeRequested = true;
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            instrument: 'XAUUSD',
            timeframe: 'M15',
            candles: window(OLDER_START, OLDER_N),
            has_more_history: false, // this page reaches the start of history
          }),
        });
      }
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          instrument: 'XAUUSD',
          timeframe: 'M15',
          candles: window(RECENT_START, RECENT_N),
          has_more_history: true, // older history exists, load on demand
        }),
      });
    });
    await page.route('**/api/market-reading**', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(readingWithOldChoch()) }),
    );
    await page.route('**/api/market-status**', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ is_open: true, state: 'open', next_open_ts: null, next_close_ts: null }) }),
    );

    await page.goto('/app?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);

    const chart = page.getByRole('application').first();
    await chart.waitFor({ state: 'visible', timeout: 60_000 });
    await page.waitForTimeout(1500);
    // The recent window is deep enough that the initial view does NOT auto-page.
    expect(beforeRequested).toBe(false);
    await chart.screenshot({ path: `${SHOTS}/07-oldevent-initial.png` });

    // The CHOCH row IS the old event (current_choch null → latestBreak = the old one).
    const choch = page.locator('.strow').filter({ hasText: 'CHOCH' }).first();
    await choch.scrollIntoViewIfNeeded();
    await choch.click();
    await expect(choch).toHaveAttribute('aria-pressed', 'true');

    // Clicking the old event triggers the backward paging (loadOlder → ?before=).
    await expect.poll(() => beforeRequested, { timeout: 10_000 }).toBe(true);
    await page.waitForTimeout(1200); // let the page land + the single camera tween settle
    await chart.screenshot({ path: `${SHOTS}/08-oldevent-framed.png` });
  });
});

test.describe('Structure events — click frames the chart (390×844)', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('event click on a narrow window frames the chart', async ({ page }) => {
    test.setTimeout(90_000);
    await mockReading(page);
    await page.goto('/app?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);

    // Stacked mobile layout: open the "Lecture" tab, expand Structure accordion.
    // Events here render via StructureSection (a <Row> with aria-pressed), NOT the
    // desktop StructureCard `.strow`; the underlying select path is identical.
    const lecture = page.getByRole('tab', { name: /Lecture/i });
    await lecture.waitFor({ state: 'visible', timeout: 60_000 });
    await lecture.click();
    const chart = page.getByRole('application').first();
    await chart.waitFor({ state: 'visible', timeout: 60_000 });
    await page.waitForTimeout(1500);
    await chart.screenshot({ path: `${SHOTS}/05-mobile-initial.png` });

    const trigger = page.getByRole('button', { name: /Structure de marché/i }).first();
    await trigger.waitFor({ state: 'visible', timeout: 60_000 });
    await trigger.scrollIntoViewIfNeeded();
    if ((await trigger.getAttribute('aria-expanded')) === 'false') await trigger.click();

    // The CHOCH event row: selectable (aria-pressed) and its value shows 2 384,20.
    const choch = page.locator('[aria-pressed]').filter({ hasText: '384,20' }).first();
    await choch.waitFor({ state: 'visible', timeout: 60_000 });
    await choch.scrollIntoViewIfNeeded();
    await choch.click();
    await expect(choch).toHaveAttribute('aria-pressed', 'true');
    await page.waitForTimeout(900);
    await chart.scrollIntoViewIfNeeded();
    await page.waitForTimeout(400);
    await chart.screenshot({ path: `${SHOTS}/06-mobile-choch-framed.png` });
  });
});
