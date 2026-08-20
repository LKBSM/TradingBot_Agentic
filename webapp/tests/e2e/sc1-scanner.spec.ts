import { expect, test, type Page } from '@playwright/test';

/**
 * SC-1 — Scanner refonte, the FIVE states at both viewports (1280×800, 390×844).
 *
 * The real scan needs a live backend (readings). Following this repo's e2e
 * convention (cf. tf1-units), we mock the access gate + the scan endpoint so the
 * five states render deterministically WITHOUT the data backend, and assert the
 * structural guarantees that must hold on every one of them: no raw i18n key, no
 * horizontal overflow, and the SC-1 invariants per state (non-maskable
 * « à l'encontre » block, isolated counts on the empty combo, non-sync mention on
 * saved readings, « zéro condition ≠ tous les marchés »). The live data path is
 * validated with the founder before merge.
 */

const CONFIG_KEY = 'mia.conditionsConfig.v1';
const STRAT_KEY = 'mia.scannerStrategies.v1';

const CTX = {
  trend: 'bearish', market_phase: 'trend', volatility_observed: 'normal',
  mtf_confluence: {}, mtf_trends: { h4: 'bearish', h1: 'bearish', m15: 'bearish' },
  bos: null, choch: null, active_order_blocks: 2, active_fair_value_gaps: 1,
  structural_range: { low: 4010, high: 4055 }, news_upcoming: [],
};

function match(over: Record<string, unknown> = {}) {
  return {
    instrument: 'XAUUSD', timeframe: 'M15', candle_close_ts: new Date().toISOString(),
    close_price: 4029, matched: true, met_count: 2, total: 2, non_evaluable_count: 0,
    conditions_met: [
      { type: 'trend_is', label: 'La tendance', met: true, detail: 'Baissière.' },
      { type: 'price_in_ob', label: 'Prix dans un OB', met: true, detail: 'OB 4028–4030.' },
    ],
    conditions_unmet: [], conditions_non_evaluable: [],
    context_against: [{ label: 'Le 4 h est en tendance haussière', detail: 'désaccord multi-unités' }],
    context: CTX,
    freshness: 'fresh', bars_behind: 0, ...over,
  };
}

async function mock(page: Page, scan: unknown) {
  await page.route('**/api/access/me', (r) =>
    r.fulfill({ json: { authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false, is_owner: true, has_access: true, subscription_required: false } }),
  );
  await page.route('**/api/conditions-scan', (r) => r.fulfill({ json: scan }));
}

async function seed(page: Page, entries: Record<string, unknown>) {
  await page.addInitScript((data: Record<string, string>) => {
    for (const [k, v] of Object.entries(data)) window.localStorage.setItem(k, v);
  }, Object.fromEntries(Object.entries(entries).map(([k, v]) => [k, JSON.stringify(v)])));
}

const RESULTS = {
  as_of: new Date().toISOString(), logic: 'AND', scanned: 3,
  matches: [
    match(),
    match({ instrument: 'EURUSD', timeframe: 'H4', matched: false, met_count: 1, total: 2,
      conditions_met: [{ type: 'trend_is', label: 'La tendance', met: true, detail: 'Baissière.' }],
      conditions_unmet: [{ type: 'price_in_ob', label: 'Prix dans un OB', met: false, detail: 'Hors OB.' }] }),
  ],
  unavailable: [],
};

const NO_COMBO = {
  as_of: new Date().toISOString(), logic: 'AND', scanned: 2,
  matches: [
    match({ matched: false, met_count: 1, total: 2,
      conditions_met: [{ type: 'trend_is', label: 'La tendance', met: true, detail: 'Baissière.' }],
      conditions_unmet: [{ type: 'price_in_ob', label: 'Prix dans un OB', met: false, detail: 'Hors OB.' }] }),
  ],
  unavailable: [],
};

const CONFIG = { logic: 'AND', conditions: [{ type: 'trend_is', trend: 'bearish' }, { type: 'price_in_ob' }] };
const STRATS = [{ id: 's1', name: 'London sweep M15', schema_version: 1, config: CONFIG, createdAt: 1, lastUsedAt: 1 }];

const RAW_KEY_SCAN = `(() => {
  const bad = []; const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); let n;
  while ((n = w.nextNode())) {
    const t = (n.textContent || '').trim();
    if (!t || t.includes('/') || t.includes('@') || t.includes(':')) continue;
    if (/^[a-z][a-zA-Z0-9]*(\\.[a-zA-Z][a-zA-Z0-9]*)+$/.test(t)) bad.push(t);
  }
  return bad;
})()`;

async function overflow(page: Page) {
  return page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
}

for (const vp of [
  { name: 'desktop 1280×800', width: 1280, height: 800 },
  { name: 'phone 390×844', width: 390, height: 844 },
]) {
  test.describe(vp.name, () => {
    test.beforeEach(async ({ page }) => {
      await page.setViewportSize({ width: vp.width, height: vp.height });
    });

    test('construction + aucune condition: families, segmented buttons, zero-note', async ({ page }) => {
      await mock(page, RESULTS);
      await page.goto('/scanner');
      await expect(page.getByText('Structure', { exact: false }).first()).toBeVisible();
      // Segmented single-choice controls (all options visible, no dropdown).
      await expect(page.getByRole('group').first()).toBeVisible();
      // No condition ticked yet → « zéro condition ne signifie pas tous les marchés ».
      await expect(page.getByText(/tous les marchés/)).toBeVisible();
      expect(await page.evaluate(RAW_KEY_SCAN)).toEqual([]);
      expect(await overflow(page)).toBeLessThanOrEqual(1);
      // Families load COLLAPSED — expand one before a condition can be ticked.
      await page.getByRole('button', { name: /Structure/ }).first().click();
      // Ticking a condition makes the sticky bar's LIVE combo count appear.
      await page.getByRole('checkbox').first().check();
      await expect(page.getByTestId('live-combo-count')).toHaveText('1', { timeout: 6000 });
    });

    test('résultats: three blocks, non-maskable « à l\'encontre », open-in-chart', async ({ page }) => {
      await seed(page, { [CONFIG_KEY]: CONFIG });
      await mock(page, RESULTS);
      await page.goto('/scanner');
      await expect(page.getByTestId('against-block').first()).toBeVisible();
      // Enriched against-signal surfaced even on a full match.
      await expect(page.getByText('Le 4 h est en tendance haussière').first()).toBeVisible();
      await expect(page.getByText('Ouvrir dans le graphique').first()).toBeVisible();
      await expect(page.getByText(/Correspondances \(/)).toBeVisible();
      await expect(page.getByText(/Trader/)).toHaveCount(0);
      expect(await page.evaluate(RAW_KEY_SCAN)).toEqual([]);
      expect(await overflow(page)).toBeLessThanOrEqual(1);
    });

    test('aucun combo: not an error, isolated counts that do not add up', async ({ page }) => {
      await seed(page, { [CONFIG_KEY]: CONFIG });
      await mock(page, NO_COMBO);
      await page.goto('/scanner');
      await expect(page.getByTestId('scan-no-combo')).toBeVisible();
      await expect(page.getByText(/Ce n’est pas une erreur/)).toBeVisible();
      await expect(page.getByText(/ne s’additionnent pas/)).toBeVisible();
      // Never proposes to loosen a condition (product line #3).
      await expect(page.getByText(/ne propose pas de/)).toBeVisible();
      expect(await overflow(page)).toBeLessThanOrEqual(1);
    });

    test('lectures enregistrées: non-sync + not-a-ranking mentions, export', async ({ page }) => {
      await seed(page, { [CONFIG_KEY]: CONFIG, [STRAT_KEY]: STRATS });
      await mock(page, RESULTS);
      await page.goto('/scanner');
      await expect(page.getByText(/ne sont pas synchronisées/)).toBeVisible();
      await expect(page.getByText(/pas un classement/)).toBeVisible();
      await expect(page.getByRole('button', { name: 'Exporter' })).toBeVisible();
      // C4: the saved reading's combo count (re-evaluated on open).
      await expect(page.getByText(/1 combo/).first()).toBeVisible({ timeout: 6000 });
      expect(await overflow(page)).toBeLessThanOrEqual(1);
    });
  });
}
