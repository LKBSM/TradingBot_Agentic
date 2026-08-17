import { test, type Page } from '@playwright/test';

/**
 * UI-2 — capture harness (NOT an assertion spec). Produces before/after
 * screenshots of the two scanner tabs (Décrire + Conditions) at both viewports
 * in fr + en, plus /app and /zones baselines to prove no shared-shell
 * regression. Output dir is driven by UI2_PHASE ("before" | "after").
 *
 * Reuses the SC-2 mocking convention (access gate + translate + scan mocked)
 * so every surface renders deterministically without the LLM/data backend.
 */

const PHASE = process.env.UI2_PHASE ?? 'before';
const DIR = `../docs/audits/ui2-shots/${PHASE}`;

const TRANSLATED = {
  outcome: 'translated',
  refusal: null,
  conditions: [
    { type: 'trend_is', trend: 'bullish' },
    { type: 'zone_untested', zone_kind: 'ob' },
    { type: 'liquidity_swept_recent', max_bars: 10 },
  ],
  assumptions: [
    {
      condition_type: 'liquidity_swept_recent',
      control: 'max_bars',
      value: '10',
      source_phrase: 'récemment',
    },
  ],
  untranslatable: [],
};

const CTX = {
  trend: 'bullish', market_phase: 'expansion', volatility_observed: 'elevated',
  mtf_confluence: {}, mtf_trends: { h4: 'bullish', h1: 'bullish', m15: 'bullish' },
  bos: null, choch: null, active_order_blocks: 2, active_fair_value_gaps: 1,
  structural_range: { low: 4010, high: 4055 }, news_upcoming: [],
};

const SCAN = {
  as_of: new Date().toISOString(), logic: 'AND', scanned: 3,
  matches: [
    {
      instrument: 'XAUUSD', timeframe: 'M15', candle_close_ts: new Date().toISOString(),
      close_price: 4029, matched: true, met_count: 3, total: 3, non_evaluable_count: 0,
      conditions_met: [
        { type: 'trend_is', label: 'La tendance', met: true, detail: 'Haussière.' },
        { type: 'zone_untested', label: 'Zone jamais testée', met: true, detail: 'OB vierge.' },
        { type: 'liquidity_swept_recent', label: 'Poche prise', met: true, detail: 'Il y a 4 bougies.' },
      ],
      conditions_unmet: [], conditions_non_evaluable: [],
      context_against: [{ label: 'Le 4 h est en tendance baissière', detail: 'désaccord multi-unités' }],
      context: CTX, freshness: 'fresh', bars_behind: 0,
    },
  ],
  unavailable: [],
};

async function mockAll(page: Page) {
  await page.route('**/api/access/me', (r) =>
    r.fulfill({
      json: {
        authenticated: true, gate_enforced: false, beta_lockdown: false,
        must_login: false, is_owner: true, has_access: true, subscription_required: false,
      },
    }),
  );
  await page.route('**/api/scanner/translate', (r) => r.fulfill({ json: TRANSLATED }));
  await page.route('**/api/conditions-scan', (r) => r.fulfill({ json: SCAN }));
  // Let /app + /zones render their empty/loading states deterministically.
  await page.route('**/api/market-reading**', (r) => r.fulfill({ status: 200, json: {} }));
  await page.route('**/api/candles**', (r) => r.fulfill({ status: 200, json: { candles: [] } }));
}

async function installFakeSpeech(page: Page) {
  await page.addInitScript(() => {
    class FakeRecognition {
      lang = ''; continuous = false; interimResults = false; maxAlternatives = 1;
      onresult: ((e: unknown) => void) | null = null;
      onerror: ((e: { error: string }) => void) | null = null;
      onend: (() => void) | null = null;
      onstart: (() => void) | null = null;
      start() { this.onstart?.(); }
      stop() { this.onend?.(); }
      abort() {}
    }
    (window as unknown as Record<string, unknown>).SpeechRecognition = FakeRecognition;
    (window as unknown as Record<string, unknown>).webkitSpeechRecognition = FakeRecognition;
  });
}

const VIEWPORTS = [
  { tag: 'desktop', width: 1280, height: 800 },
  { tag: 'mobile', width: 390, height: 844 },
] as const;

const LOCALES = ['fr', 'en'] as const;

async function shoot(page: Page, name: string) {
  await page.screenshot({ path: `${DIR}/${name}.png`, fullPage: true });
}

// Single project run (chromium-desktop) drives everything; we set the viewport
// per-shot so both sizes come out of one worker deterministically.
test('UI-2 captures', async ({ page }) => {
  test.setTimeout(180_000);
  await installFakeSpeech(page);
  await mockAll(page);

  for (const loc of LOCALES) {
    for (const vp of VIEWPORTS) {
      await page.setViewportSize({ width: vp.width, height: vp.height });

      // Describe tab — initial state.
      await page.goto(`/${loc}/scanner/decrire`);
      await page.getByTestId('describe-input').waitFor({ state: 'visible' });
      await page.waitForTimeout(400);
      await shoot(page, `scanner-decrire-initial_${loc}_${vp.tag}`);

      // Describe tab — translated cards state.
      await page.getByTestId('describe-input').fill(
        loc === 'fr'
          ? 'OB vierge en tendance haussière, poche prise récemment'
          : 'untested OB in a bullish trend, liquidity swept recently',
      );
      await page.getByTestId('translate-button').click();
      await page.getByTestId('translated-card').first().waitFor({ state: 'visible' });
      await page.waitForTimeout(300);
      await shoot(page, `scanner-decrire-translated_${loc}_${vp.tag}`);

      // Conditions tab (manual palette / builder onboarding).
      await page.goto(`/${loc}/scanner`);
      await page.waitForTimeout(600);
      await shoot(page, `scanner-conditions_${loc}_${vp.tag}`);

      // Shared-shell non-regression baselines.
      await page.goto(`/${loc}/app`);
      await page.waitForTimeout(600);
      await shoot(page, `app_${loc}_${vp.tag}`);

      await page.goto(`/${loc}/zones`);
      await page.waitForTimeout(600);
      await shoot(page, `zones_${loc}_${vp.tag}`);
    }
  }
});
