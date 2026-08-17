import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * VZ-1 /zones — proximity, confluence, the contact ledger, the « Comblées »
 * group and the M.I.A panel. Backend-free: the market-reading + candles endpoints
 * are mocked (PERF-1 pattern). The reading is crafted so ONE zone has the price
 * inside, ONE sits above with contacts, ONE is untouched with NO confluence, and
 * ONE is filled (consumed). Sibling timeframes return a payload that overlaps the
 * first two zones but NOT the untouched one, so its confluence stays empty.
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
function fvg(over: Record<string, unknown>) {
  return {
    id: 'x', direction: 'bullish', level_high: 0, level_low: 0, status: 'active',
    created_at: iso(8), tested: false, user_flagged: false, contacts: [], ...over,
  };
}

// The M15 reading: the four scenario zones.
const READING = {
  ...FIXTURE_XAU_M15,
  header: { ...FIXTURE_XAU_M15.header, close_price: PRICE },
  structure: {
    ...FIXTURE_XAU_M15.structure,
    order_blocks: [
      ob({
        id: 'ob-inside', level_low: 2388, level_high: 2392, tested: true,
        contacts: [{ at: iso(16, 40), level: 2390, outcome: 'inside' }],
        origin: { kind: 'bos', direction: 'bullish', at: iso(9), level: 2386 },
      }),
      ob({ id: 'ob-untouched', level_low: 2350, level_high: 2352 }),
    ],
    fair_value_gaps: [
      fvg({
        id: 'fvg-above', level_low: 2400, level_high: 2405, status: 'partially_filled',
        tested: true, fill_level: 2402,
        contacts: [
          { at: iso(10, 30), level: 2402, outcome: 'entry_exit' },
          { at: iso(15), level: 2404.9, outcome: 'edge_touch' },
        ],
      }),
    ],
    consumed_order_blocks: [],
    consumed_fair_value_gaps: [
      fvg({
        id: 'fvg-filled', level_low: 2360, level_high: 2364, status: 'filled', tested: true,
        contacts: [
          { at: iso(3), level: 2362, outcome: 'entry_exit' },
          { at: iso(8, 15), level: 2360, outcome: 'traversal' },
        ],
      }),
    ],
    liquidity_pools: [
      { id: 'liq-1', side: 'bsl', kind: 'equal_highs', level: 2406, touches: 2, is_external: true, status: 'intact', created_at: iso(7), user_flagged: false },
    ],
  },
};

// Sibling timeframes: overlap the inside/above zones but NOT the untouched one.
const SIBLING = {
  ...READING,
  structure: {
    ...READING.structure,
    order_blocks: [ob({ id: 'ob-h1-wrap', level_low: 2387, level_high: 2393 })],
    fair_value_gaps: [fvg({ id: 'fvg-h1-wrap', level_low: 2399, level_high: 2406 })],
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

async function mock(page: Page, reading = READING) {
  await page.route('**/api/candles**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(candles()) }),
  );
  await page.route('**/api/market-reading**', (route) => {
    const url = route.request().url();
    const body = /timeframe=M15(\b|&|$)/.test(url) ? reading : SIBLING;
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
  });
}

async function openZones(page: Page, reading = READING) {
  test.setTimeout(90_000);
  await mock(page, reading);
  await page.goto('/zones?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
  await dismissCookieBanner(page);
  await page.locator('[data-zone-id="ob-inside"]').waitFor({ state: 'visible', timeout: 60_000 });
}

function scenarios() {
  test('groups the zones by position, incl. the « Comblées » group', async ({ page }) => {
    await openZones(page);
    // Scope to the group separators (`.zsep`) — the M.I.A subject line echoes the
    // same wording in lowercase, so a bare getByText would be ambiguous.
    await expect(page.locator('.zsep', { hasText: 'Le prix est dedans' })).toBeVisible();
    await expect(page.locator('.zsep', { hasText: 'Au-dessus du prix' })).toBeVisible();
    await expect(page.locator('.zsep', { hasText: 'Sous le prix' })).toBeVisible();
    await expect(page.locator('.zsep', { hasText: 'Comblées' })).toBeVisible();
  });

  test('the inside zone shows the inside-proximity block + « Jamais comblée »', async ({ page }) => {
    await openZones(page);
    const card = page.locator('[data-zone-id="ob-inside"]');
    await expect(card.locator('.zpx.inside')).toBeVisible();
    await expect(card).toContainText('à l’intérieur');
    await expect(card).toContainText('Jamais comblée');
  });

  test('the above zone shows a contact ledger with distinct outcomes', async ({ page }) => {
    await openZones(page);
    const card = page.locator('[data-zone-id="fvg-above"]');
    // Compact card: the contact STATE and the distance (with unit + edge) show
    // up front (VZ-2 hierarchy levels 2/3).
    await expect(card).toContainText('2 contacts');
    await expect(card).toContainText('mesuré au bord');
    // The full per-contact ledger is deferred to « Détails » (VZ-2 density) — one
    // click reveals both distinct outcomes, never merged.
    await card.locator('.zdeth').click();
    await expect(card).toContainText('est entré à');
    await expect(card).toContainText('touché le bord sans y pénétrer');
  });

  test('the untouched zone shows the explicit no-confluence absence state', async ({ page }) => {
    await openZones(page);
    const card = page.locator('[data-zone-id="ob-untouched"]');
    await expect(card).toContainText('Jamais touchée');
    // Confluence (incl. the explicit absence state) is deferred to « Détails »
    // (VZ-2 density) — one click, never a silent implication of presence.
    await card.locator('.zdeth').click();
    await expect(card.locator('.zconf.none')).toBeVisible();
    await expect(card).toContainText('Rien d’autre n’est détecté');
  });

  test('an empty filter states it explicitly and never suggests relaxing it', async ({ page }) => {
    // A reading with NO consumed zones → the « Comblées » filter is empty.
    const noConsumed = {
      ...READING,
      structure: { ...READING.structure, consumed_fair_value_gaps: [], consumed_order_blocks: [] },
    };
    await openZones(page, noConsumed);
    await page.getByRole('button', { name: 'Comblées' }).click();
    const empty = page.getByTestId('zones-empty');
    await expect(empty).toBeVisible();
    await expect(empty).toContainText(/aucune zone comblée/i);
    await expect(empty).not.toContainText(/assoupl|élargir|relax|moins strict/i);
  });

  test('never renders « chevauche » nor judgement wording', async ({ page }) => {
    await openZones(page);
    const body = await page.locator('.pagewrap').innerText();
    expect(body).not.toMatch(/chevauche/i);
    expect(body).not.toMatch(/respect|valid|solide|fiable|qualité|meilleur/i);
  });
}

test.describe('VZ-1 /zones @ 1280×800', () => {
  test.use({ viewport: { width: 1280, height: 800 } });
  scenarios();

  test('the M.I.A panel switches subject on a card click (no reload)', async ({ page }) => {
    await openZones(page);
    const subject = page.getByTestId('mia-subject').first();
    await expect(subject).toContainText('388,00'); // default = the inside zone
    await page.locator('[data-zone-id="ob-untouched"]').click();
    await expect(subject).toContainText('350,00'); // switched, same page
  });

  test('the M.I.A free-text input answers locally (no LLM)', async ({ page }) => {
    await openZones(page);
    const input = page.getByPlaceholder('Pose ta question sur cette zone…').first();
    await input.fill("qu'est-ce qu'il y a d'autre à ce niveau");
    await input.press('Enter');
    await expect(
      page.locator('.zmia-body .bub.a').last(),
    ).toContainText(/au même niveau|rien d’autre n’est détecté|poche de liquidité|à l’intérieur|englobe/i);
  });
});

test.describe('VZ-1 /zones @ 390×844', () => {
  test.use({ viewport: { width: 390, height: 844 } });
  scenarios();

  test('the M.I.A panel is a bottom sheet opened by a button', async ({ page }) => {
    await openZones(page);
    await page.getByRole('button', { name: 'Demander à M.I.A' }).click();
    const sheet = page.locator('.zmia-sheet');
    await expect(sheet).toBeVisible();
    await expect(sheet.getByTestId('mia-subject')).toContainText('388,00');
  });
});
