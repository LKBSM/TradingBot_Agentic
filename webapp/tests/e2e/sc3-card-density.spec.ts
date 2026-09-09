import { expect, test, type Page } from '@playwright/test';

/**
 * SC-3 — the scanner card's density pass, at 1280×800 and 390×844.
 *
 * Same mocking convention as sc1-scanner.spec: the access gate and the scan
 * endpoint are stubbed so the card renders deterministically without the data
 * backend. What is asserted here is exactly what a screenshot cannot prove:
 *   · the chip row wraps and never pushes the page into horizontal scroll;
 *   · every chip dot is actually PAINTED (the OB/FVG theme tokens are 4–13 %
 *     alpha zone fills — used raw they would be invisible), and OB ≠ FVG in
 *     shape so they stay distinguishable even where a theme paints both white;
 *   · block 3 folds and unfolds, at BOTH viewports;
 *   · block 2 is visible without any interaction, and is in no <details>;
 *   · one legal disclaimer per page, still.
 */

const CONFIG_KEY = 'mia.conditionsConfig.v1';

const CTX = {
  trend: 'bullish',
  market_phase: 'expansion',
  volatility_observed: 'normal',
  mtf_confluence: {},
  mtf_trends: { h4: 'bearish', h1: 'bearish', m15: 'bearish' },
  bos: { direction: 'bearish', level: 4380, validation_status: 'confirmed' },
  choch: null,
  active_order_blocks: 4,
  active_fair_value_gaps: 10,
  structural_range: { low: 4224.23, high: 4695.94 },
  news_upcoming: [
    { event: 'US NFP', impact: 'high', time_to_event_min: 30 },
    { event: 'US CPI', impact: 'high', time_to_event_min: 90 },
  ],
};

/** A combo with NOTHING optional: no break, no range, no active zone. */
const CTX_BARE = {
  ...CTX,
  bos: null,
  choch: null,
  active_order_blocks: 0,
  active_fair_value_gaps: 0,
  structural_range: null,
  news_upcoming: [],
};

function match(over: Record<string, unknown> = {}) {
  return {
    instrument: 'XAUUSD',
    timeframe: 'H4',
    candle_close_ts: new Date().toISOString(),
    close_price: 4408.91702,
    matched: true,
    met_count: 2,
    total: 2,
    non_evaluable_count: 0,
    conditions_met: [
      { type: 'trend_is', label: 'La tendance structurelle est', met: true, detail: 'haussier.' },
      { type: 'market_phase_is', label: 'La phase de marché est', met: true, detail: 'expansion.' },
    ],
    conditions_unmet: [],
    conditions_non_evaluable: [],
    context_against: [
      { label: 'Le 5 min, 15 min, 1 h', detail: 'en tendance baissière — désaccord multi-unités' },
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
  scanned: 2,
  matches: [match(), match({ instrument: 'EURUSD', timeframe: 'D1', context: CTX_BARE })],
  unavailable: [],
};

const CONFIG = {
  logic: 'AND',
  conditions: [{ type: 'trend_is', trend: 'bullish' }, { type: 'market_phase_is', phase: 'expansion' }],
};

async function mock(page: Page) {
  await page.route('**/api/access/me', (r) =>
    r.fulfill({
      json: {
        authenticated: true, gate_enforced: false, beta_lockdown: false,
        must_login: false, is_owner: true, has_access: true, subscription_required: false,
      },
    }),
  );
  await page.route('**/api/conditions-scan', (r) => r.fulfill({ json: RESULTS }));
  await page.addInitScript(
    ([k, v]: [string, string]) => window.localStorage.setItem(k, v),
    [CONFIG_KEY, JSON.stringify(CONFIG)] as [string, string],
  );
}

async function overflow(page: Page) {
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
}

for (const vp of [
  { name: 'desktop 1280×800', width: 1280, height: 800 },
  { name: 'phone 390×844', width: 390, height: 844 },
]) {
  test.describe(vp.name, () => {
    test.beforeEach(async ({ page }) => {
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await mock(page);
      await page.goto('/scanner');
      await expect(page.locator('.combo').first()).toBeVisible();
    });

    test('the chip row wraps inside its card and never scrolls the page sideways', async ({ page }) => {
      const chips = page.locator('.combo').first().locator('.chips');
      await expect(chips).toBeVisible();
      expect(await chips.locator('.chip').count()).toBe(6);

      // Every chip stays within its card's content box (wrapping, not clipping).
      const fits = await page.evaluate(() => {
        const card = document.querySelector('.combo');
        const row = card?.querySelector('.chips');
        if (!card || !row) return null;
        const cb = card.getBoundingClientRect();
        return Array.from(row.querySelectorAll('.chip')).every((c) => {
          const r = c.getBoundingClientRect();
          return r.left >= cb.left - 1 && r.right <= cb.right + 1;
        });
      });
      expect(fits).toBe(true);
      expect(await overflow(page)).toBeLessThanOrEqual(1);

      // …and the card it wraps inside is actually wide enough to read. The grid
      // had NO breakpoint: at 390px it still cut two columns, measured at 181px
      // and 158px. One column on phones, two from 768px up.
      const tracks = await page
        .locator('.resgrid')
        .first()
        .evaluate((el) => getComputedStyle(el).gridTemplateColumns.split(/\s+/).length);
      expect(tracks).toBe(vp.width < 768 ? 1 : 2);
      const cardWidth = await page
        .locator('.combo')
        .first()
        .evaluate((el) => Math.round(el.getBoundingClientRect().width));
      expect(cardWidth).toBeGreaterThan(300);
    });

    test('every chip dot is actually painted, and OB differs from FVG in shape', async ({ page }) => {
      const dots = await page.evaluate(() => {
        const row = document.querySelector('.combo .chips');
        return Array.from(row?.querySelectorAll('.chip .dot') ?? []).map((d) => {
          const s = getComputedStyle(d as Element);
          const r = (d as Element).getBoundingClientRect();
          return {
            cls: (d as Element).className,
            bgColor: s.backgroundColor,
            bgImage: s.backgroundImage,
            radius: s.borderTopLeftRadius,
            w: Math.round(r.width),
            h: Math.round(r.height),
          };
        });
      });
      expect(dots.length).toBe(6);
      for (const d of dots) {
        expect(d.w).toBeGreaterThan(0);
        expect(d.h).toBeGreaterThan(0);
        // Painted: either an opaque background colour, or the opacifying layer.
        const painted = d.bgImage !== 'none' || !/rgba\(0, 0, 0, 0\)|transparent/.test(d.bgColor);
        expect(painted, `dot ${d.cls} is not painted`).toBe(true);
      }
      const ob = dots.find((d) => d.cls.includes('d-ob'));
      const fvg = dots.find((d) => d.cls.includes('d-fvg'));
      // Both composite their translucent token over an opaque ground…
      expect(ob?.bgImage).not.toBe('none');
      expect(fvg?.bgImage).not.toBe('none');
      // …and are told apart by SHAPE, which survives a theme painting both white.
      expect(ob?.radius).not.toBe(fvg?.radius);
    });

    test('block 3 is folded by default and opens on click', async ({ page }) => {
      const card = page.locator('.combo').first();
      const details = card.getByTestId('context-block');
      await expect(details).toHaveJSProperty('open', false);
      // Folded: the body is not visible, but the news warning still reads.
      await expect(card.locator('.ctx2')).toBeHidden();
      await expect(details.locator('summary')).toContainText('2 actus importantes');
      await details.locator('summary').click();
      await expect(details).toHaveJSProperty('open', true);
      await expect(card.locator('.ctx2')).toBeVisible();
      await expect(card.locator('.ctx2')).toContainText('4224.23–4695.94');
      expect(await overflow(page)).toBeLessThanOrEqual(1);
    });

    test('block 2 needs no click and sits in no <details>', async ({ page }) => {
      const against = page.getByTestId('against-block').first();
      await expect(against).toBeVisible();
      await expect(
        page.getByText('en tendance baissière — désaccord multi-unités').first(),
      ).toBeVisible();
      const inDetails = await against.evaluate((el) => el.closest('details') !== null);
      expect(inDetails).toBe(false);
    });

    test('a combo with no break, no range and no active zone shows no empty chip', async ({ page }) => {
      const bare = page.locator('.combo', { hasText: 'EUR/USD' }).first();
      const keys = await bare
        .locator('.chips .chip')
        .evaluateAll((els) => els.map((e) => e.getAttribute('data-chip')));
      expect(keys).toEqual(['trend', 'phase']);
      // Never a dash and never a defaulted zero in place of a missing field.
      await expect(bare.locator('.chips')).not.toContainText('0 OB');
      await expect(bare.locator('.chips')).not.toContainText('0 FVG');
    });

    test('still one legal disclaimer on the page, and no forbidden « cible » on a card', async ({ page }) => {
      // Scoped to the RESULT CARDS: the page as a whole legitimately says
      // « …pas de cible » in the palette note — a DENIAL of the forbidden thing
      // is the promise the product makes, not a use of it. What must never come
      // back is the positive form the engine used to emit, « (cible : … ) ».
      const cards = (await page.locator('.combo').allTextContents()).join(' ').toLowerCase();
      expect(cards).not.toMatch(/\bcible\b/);
      const main = (await page.locator('main, .pagewrap').first().textContent()) ?? '';
      expect(main.toLowerCase()).not.toMatch(/\(\s*cible\s*:/);

      // One VISIBLE disclaimer, the CLN-1 §5 rule. Two nodes exist by design —
      // the rail footer and the mobile footer — and CSS shows exactly one at any
      // width; counting DOM nodes would wrongly report two. Same method as
      // cln-1-disclaimers.spec, and adding a card-level notice back would fail
      // it here too.
      const notices = page.locator('p', { hasText: 'Lecture algorithmique éducative' });
      let visible = 0;
      for (const el of await notices.all()) if (await el.isVisible()) visible += 1;
      expect(visible).toBe(1);
    });

    test('the toolbar carries no auto-refresh switch any more', async ({ page }) => {
      await expect(page.getByRole('switch')).toHaveCount(0);
      await expect(page.getByText(/Actualisation auto/i)).toHaveCount(0);
      await expect(page.getByRole('button', { name: /Relancer le scan/ })).toBeVisible();
    });
  });
}
