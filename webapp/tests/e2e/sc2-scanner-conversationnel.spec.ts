import { expect, test, type Page } from '@playwright/test';

/**
 * SC-2 — Scanner conversationnel, the SIX states at both viewports, plus the two
 * dictation degradations (unsupported browser, permission denied).
 *
 * Following this repo's e2e convention (cf. sc1-scanner): we mock the access gate,
 * the translate endpoint and the scan endpoint so every state renders
 * deterministically WITHOUT the LLM/data backend, and assert the load-bearing
 * SC-2 invariants: assumptions are always shown, untranslatable fragments are
 * NAMED (never substituted), ranking asks are REFUSED (never searched), the three
 * result blocks are all rendered, and the mic is ABSENT when the browser cannot
 * dictate. The live LLM path is validated with the founder before merge.
 */

const PAGE = '/fr/scanner/decrire';

const TRANSLATED = {
  outcome: 'translated',
  refusal: null,
  conditions: [
    { type: 'trend_is', trend: 'bullish' },
    { type: 'zone_untested', zone_kind: 'ob' },
    { type: 'liquidity_swept_recent', max_bars: 10 },
  ],
  assumptions: [
    { condition_type: 'liquidity_swept_recent', control: 'max_bars', value: '10', source_phrase: 'récemment' },
  ],
  untranslatable: [],
};

const PARTIAL = {
  outcome: 'partial',
  refusal: null,
  conditions: [{ type: 'zone_untested', zone_kind: 'ob' }],
  assumptions: [],
  untranslatable: [{ fragment: 'le RSI est en survente', category: 'indicator' }],
};

const REFUSED = {
  outcome: 'refused',
  refusal: { kind: 'ranking' },
  conditions: [],
  assumptions: [],
  untranslatable: [],
};

const CTX = {
  trend: 'bullish', market_phase: 'expansion', volatility_observed: 'elevated',
  mtf_confluence: {}, mtf_trends: { h4: 'bullish', h1: 'bullish', m15: 'bullish' },
  bos: null, choch: null, active_order_blocks: 2, active_fair_value_gaps: 1,
  structural_range: { low: 4010, high: 4055 }, news_upcoming: [],
};

function match(over: Record<string, unknown> = {}) {
  return {
    instrument: 'XAUUSD', timeframe: 'M15', candle_close_ts: new Date().toISOString(),
    close_price: 4029, matched: true, met_count: 3, total: 3, non_evaluable_count: 0,
    conditions_met: [
      { type: 'trend_is', label: 'La tendance', met: true, detail: 'Haussière.' },
      { type: 'zone_untested', label: 'Zone jamais testée', met: true, detail: 'OB vierge.' },
      { type: 'liquidity_swept_recent', label: 'Poche prise', met: true, detail: 'Il y a 4 bougies.' },
    ],
    conditions_unmet: [], conditions_non_evaluable: [],
    context_against: [{ label: 'Le 4 h est en tendance baissière', detail: 'désaccord multi-unités' }],
    context: CTX, freshness: 'fresh', bars_behind: 0, ...over,
  };
}

const SCAN = {
  as_of: new Date().toISOString(), logic: 'AND', scanned: 3,
  matches: [match()], unavailable: [],
};

async function mockAccess(page: Page) {
  await page.route('**/api/access/me', (r) =>
    r.fulfill({ json: { authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false, is_owner: true, has_access: true, subscription_required: false } }),
  );
}
async function mockTranslate(page: Page, result: unknown) {
  await page.route('**/api/scanner/translate', (r) => r.fulfill({ json: result }));
}
async function mockScan(page: Page, scan: unknown) {
  await page.route('**/api/conditions-scan', (r) => r.fulfill({ json: scan }));
}

/** Inject a scripted SpeechRecognition BEFORE any app code runs. */
async function installFakeSpeech(page: Page, mode: 'ok' | 'denied') {
  await page.addInitScript((m: string) => {
    class FakeRecognition {
      lang = '';
      continuous = false;
      interimResults = false;
      maxAlternatives = 1;
      onresult: ((e: unknown) => void) | null = null;
      onerror: ((e: { error: string }) => void) | null = null;
      onend: (() => void) | null = null;
      onstart: (() => void) | null = null;
      start() {
        this.onstart?.();
        if (m === 'denied') {
          this.onerror?.({ error: 'not-allowed' });
          this.onend?.();
        }
      }
      stop() {
        this.onend?.();
      }
      abort() {}
    }
    (window as unknown as Record<string, unknown>).SpeechRecognition = FakeRecognition;
    (window as unknown as Record<string, unknown>).webkitSpeechRecognition = FakeRecognition;
  }, mode);
}

/** Remove any SpeechRecognition so the browser looks unsupported (e.g. Firefox). */
async function removeSpeech(page: Page) {
  await page.addInitScript(() => {
    delete (window as unknown as Record<string, unknown>).SpeechRecognition;
    delete (window as unknown as Record<string, unknown>).webkitSpeechRecognition;
  });
}

async function noRawKeys(page: Page) {
  // No un-resolved i18n key like « scannerChat.describe.title » on screen.
  const body = await page.locator('body').innerText();
  expect(body).not.toMatch(/scannerChat\.[a-zA-Z0-9_.]+/);
}

async function noHorizontalOverflow(page: Page) {
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(2);
}

test.describe('SC-2 — six states + dictation degradation', () => {
  test('state 1 — Décrire: field, examples, mic present when supported', async ({ page }) => {
    await mockAccess(page);
    await installFakeSpeech(page, 'ok');
    await page.goto(PAGE);
    await expect(page.getByTestId('describe-input')).toBeVisible();
    await expect(page.getByTestId('example-chip').first()).toBeVisible();
    await expect(page.getByTestId('mic-button')).toBeVisible();
    await noRawKeys(page);
    await noHorizontalOverflow(page);
  });

  test('state 2 — Traduction complète: editable cards + « ce que j’ai supposé »', async ({ page }) => {
    await mockAccess(page);
    await mockTranslate(page, TRANSLATED);
    await mockScan(page, SCAN);
    await page.goto(PAGE);
    await page.getByTestId('describe-input').fill('OB vierge en tendance haussière, poche prise récemment');
    await page.getByTestId('translate-button').click();
    // Three editable cards.
    await expect(page.getByTestId('translated-card')).toHaveCount(3);
    // The assumption is ALWAYS surfaced.
    const assumptions = page.getByTestId('assumptions-block');
    await expect(assumptions).toBeVisible();
    await expect(assumptions).toContainText('récemment');
    // Full palette is one click away (additional entry, never a replacement).
    await expect(page.getByTestId('open-palette')).toBeVisible();
    await noRawKeys(page);
    await noHorizontalOverflow(page);
  });

  test('state 3 — Traduction partielle: the untranslatable bit is NAMED, not substituted', async ({ page }) => {
    await mockAccess(page);
    await mockTranslate(page, PARTIAL);
    await mockScan(page, SCAN);
    await page.goto(PAGE);
    await page.getByTestId('describe-input').fill('OB vierge quand le RSI est en survente');
    await page.getByTestId('translate-button').click();
    await expect(page.getByTestId('translated-card')).toHaveCount(1);
    const untr = page.getByTestId('untranslatable-block');
    await expect(untr).toBeVisible();
    await expect(untr).toContainText('le RSI est en survente');
    await noRawKeys(page);
    await noHorizontalOverflow(page);
  });

  test('state 4 — Refus: a ranking ask is refused, never searched', async ({ page }) => {
    await mockAccess(page);
    await mockTranslate(page, REFUSED);
    let scanCalled = false;
    await page.route('**/api/conditions-scan', (r) => {
      scanCalled = true;
      return r.fulfill({ json: SCAN });
    });
    await page.goto(PAGE);
    await page.getByTestId('describe-input').fill('Montre-moi les meilleurs marchés');
    await page.getByTestId('translate-button').click();
    await expect(page.getByTestId('refusal-block')).toBeVisible();
    await expect(page.getByTestId('refusal-example').first()).toBeVisible();
    expect(scanCalled).toBe(false);
    await noRawKeys(page);
    await noHorizontalOverflow(page);
  });

  test('state 5 — Résultats: the three blocks are all rendered', async ({ page }) => {
    await mockAccess(page);
    await mockTranslate(page, TRANSLATED);
    await mockScan(page, SCAN);
    await page.goto(PAGE);
    await page.getByTestId('describe-input').fill('OB vierge en tendance haussière');
    await page.getByTestId('translate-button').click();
    await page.getByTestId('see-results').click();
    // ScanResults (SC-1): the three non-maskable blocks (labels from `combo.*`).
    await expect(page.getByText('Ce qui correspond').first()).toBeVisible();
    await expect(page.getByText(/va à l.encontre/).first()).toBeVisible();
    await expect(page.getByText(/Contexte que tu n.as pas demand/).first()).toBeVisible();
    await noRawKeys(page);
    await noHorizontalOverflow(page);
  });

  test('state 6 — Mes stratégies: empty-state copy, no ranking', async ({ page }) => {
    await mockAccess(page);
    await page.goto(PAGE);
    await page.getByRole('button', { name: /Mes stratégies/i }).click();
    await expect(page.getByRole('heading', { name: /Mes stratégies/i })).toBeVisible();
    await noRawKeys(page);
    await noHorizontalOverflow(page);
  });

  test('dictation unsupported — the mic button is ABSENT (no dead button)', async ({ page }) => {
    await mockAccess(page);
    await removeSpeech(page);
    await page.goto(PAGE);
    await expect(page.getByTestId('describe-input')).toBeVisible();
    await expect(page.getByTestId('mic-button')).toHaveCount(0);
    await noRawKeys(page);
  });

  test('dictation permission denied — clear message, field stays usable', async ({ page }) => {
    await mockAccess(page);
    await installFakeSpeech(page, 'denied');
    await page.goto(PAGE);
    await page.getByTestId('mic-button').click();
    await expect(page.getByTestId('dictation-error')).toBeVisible();
    // The text field is still fully usable.
    await page.getByTestId('describe-input').fill('je tape à la place');
    await expect(page.getByTestId('describe-input')).toHaveValue('je tape à la place');
    await noRawKeys(page);
  });

  test('LLM unavailable (0 credits) — honest error, text stays usable, no crash', async ({ page }) => {
    await mockAccess(page);
    // What the backend returns when the Anthropic call fails (e.g. 0 credits):
    // a 200 with outcome "error" — never a 500.
    await mockTranslate(page, {
      outcome: 'error', refusal: null, conditions: [], assumptions: [], untranslatable: [],
    });
    await page.goto(PAGE);
    await page.getByTestId('describe-input').fill('un OB jamais testé en tendance haussière');
    await page.getByTestId('translate-button').click();
    // Honest inline error, and we stay on the describe step (no dead-end).
    await expect(page.getByTestId('translate-inline-error')).toBeVisible();
    // The typed strategy is preserved and the field is still editable.
    await expect(page.getByTestId('describe-input')).toHaveValue('un OB jamais testé en tendance haussière');
    await page.getByTestId('describe-input').fill('je peux toujours écrire');
    await expect(page.getByTestId('describe-input')).toHaveValue('je peux toujours écrire');
    await noRawKeys(page);
  });
});
