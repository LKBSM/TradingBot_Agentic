import { expect, test, type Page } from '@playwright/test';

/**
 * SC-4 — traduction et recherche EN DIRECT pendant la saisie.
 *
 * Same convention as sc1/sc2: the access gate, the translate endpoint and the
 * scan endpoint are mocked, so the trigger policy and the reconciliation are
 * what is under test — not the LLM or the data backend. The live LLM path is
 * confirmed visually with the founder before merge.
 *
 * The load-bearing invariants asserted here:
 *   · a reading appears from TYPING ALONE — no click, and no « Voir les
 *     résultats » button left to press;
 *   · « Ce qui va à l'encontre » is rendered in the live flow exactly as in the
 *     click flow: present, and NEVER inside a <details> (not even one open by
 *     default), at BOTH viewports;
 *   · a condition removed by hand is not silently re-added while the words that
 *     produced it stand — and DOES come back once they are rewritten;
 *   · the three adversarial gestures the mission named behave as documented.
 */

const PAGE = '/fr/scanner/decrire';
const DESKTOP = { width: 1280, height: 800 };
const MOBILE = { width: 390, height: 844 };

const SENTENCE = 'Un Order Block jamais testé en tendance haussière';

/** Two conditions, each attributed to a real fragment of SENTENCE. */
const TRANSLATED = {
  outcome: 'translated',
  refusal: null,
  conditions: [
    { type: 'trend_is', trend: 'bullish' },
    { type: 'zone_untested', zone_kind: 'ob' },
  ],
  condition_sources: ['tendance haussière', 'jamais testé'],
  assumptions: [],
  untranslatable: [],
};

/** The same two conditions, the second now attributed to NEW words. */
const REDESCRIBED = {
  ...TRANSLATED,
  condition_sources: ['tendance haussière', 'zone vierge'],
};

const REFUSED = {
  outcome: 'refused',
  refusal: { kind: 'ranking' },
  conditions: [],
  condition_sources: [],
  assumptions: [],
  untranslatable: [],
};

const CTX = {
  trend: 'bullish',
  market_phase: 'expansion',
  volatility_observed: 'elevated',
  mtf_confluence: {},
  mtf_trends: { h4: 'bearish', h1: 'bullish', m15: 'bullish' },
  bos: null,
  choch: null,
  active_order_blocks: 2,
  active_fair_value_gaps: 1,
  structural_range: { low: 4010, high: 4055 },
  news_upcoming: [],
};

const SCAN = {
  as_of: new Date().toISOString(),
  logic: 'AND',
  scanned: 10,
  matches: [
    {
      instrument: 'XAUUSD',
      timeframe: 'M15',
      candle_close_ts: new Date().toISOString(),
      close_price: 4029,
      matched: false,
      met_count: 1,
      total: 2,
      non_evaluable_count: 0,
      conditions_met: [
        { type: 'trend_is', label: 'La tendance structurelle est', met: true, detail: 'haussière.' },
      ],
      // Both halves of block 2 are populated: an unmet condition AND a factual
      // against-signal. Neither may end up behind a disclosure.
      conditions_unmet: [
        { type: 'zone_untested', label: 'Zone jamais testée', met: false, detail: 'Déjà testée 2 fois.' },
      ],
      conditions_non_evaluable: [],
      context_against: [
        { label: 'Le 4 h est en tendance baissière', detail: 'désaccord multi-unités' },
      ],
      context: CTX,
      freshness: 'fresh',
      bars_behind: 0,
    },
  ],
  unavailable: [],
};

async function mockAccess(page: Page) {
  await page.route('**/api/access/me', (r) =>
    r.fulfill({
      json: {
        authenticated: true,
        gate_enforced: false,
        beta_lockdown: false,
        must_login: false,
        is_owner: true,
        has_access: true,
        subscription_required: false,
      },
    }),
  );
}

/** Mock /translate and count the paid calls it would have cost. */
async function mockTranslate(page: Page, result: unknown) {
  const calls: string[] = [];
  await page.route('**/api/scanner/translate', async (r) => {
    const body = r.request().postDataJSON() as { text?: string };
    calls.push(body?.text ?? '');
    await r.fulfill({ json: result });
  });
  return calls;
}

async function mockScan(page: Page, scan: unknown = SCAN) {
  await page.route('**/api/conditions-scan', (r) => r.fulfill({ json: scan }));
}

async function noRawKeys(page: Page) {
  const body = await page.locator('body').innerText();
  expect(body).not.toMatch(/scannerChat\.[a-zA-Z0-9_.]+/);
}

async function noHorizontalOverflow(page: Page) {
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(2);
}

/** Type the way a person does, so the debounce sees real pauses. */
async function typeSentence(page: Page, value: string) {
  await page.getByTestId('describe-input').pressSequentially(value, { delay: 12 });
}

for (const [name, viewport] of [
  ['1280x800', DESKTOP],
  ['390x844', MOBILE],
] as const) {
  test.describe(`SC-4 @ ${name}`, () => {
    test.use({ viewport });

    test('a reading and its results appear from typing alone', async ({ page }) => {
      await mockAccess(page);
      await mockTranslate(page, TRANSLATED);
      await mockScan(page);
      await page.goto(PAGE);

      await typeSentence(page, SENTENCE);

      // No click anywhere in this test.
      await expect(page.getByTestId('translated-card')).toHaveCount(2, { timeout: 10_000 });
      await expect(page.getByTestId('scan-freshness')).toBeVisible({ timeout: 10_000 });

      // The second click is gone, not merely bypassed.
      await expect(page.getByTestId('see-results')).toHaveCount(0);

      await noRawKeys(page);
      await noHorizontalOverflow(page);
    });

    test('« Ce qui va à l’encontre » is present and NOT collapsible in the live flow', async ({
      page,
    }) => {
      await mockAccess(page);
      await mockTranslate(page, TRANSLATED);
      await mockScan(page);
      await page.goto(PAGE);
      await typeSentence(page, SENTENCE);

      const against = page.getByTestId('against-block');
      await expect(against).toBeVisible({ timeout: 10_000 });

      // (1) It is not inside a <details> — not even one open by default.
      const insideDisclosure = await against.evaluate((el) => !!el.closest('details'));
      expect(insideDisclosure).toBe(false);

      // (2) Its CONTENT is on screen without any interaction: the unmet
      // condition and the factual against-signal, both readable now.
      const card = page.locator('.combo').first();
      await expect(card).toContainText('Zone jamais testée');
      await expect(card).toContainText('Le 4 h est en tendance baissière');

      // (3) Nothing in the card hides it behind a toggle.
      const hiddenAncestor = await against.evaluate((el) => {
        let node: HTMLElement | null = el as HTMLElement;
        while (node) {
          const style = getComputedStyle(node);
          if (style.display === 'none' || style.visibility === 'hidden') return true;
          node = node.parentElement;
        }
        return false;
      });
      expect(hiddenAncestor).toBe(false);
    });
  });
}

test.describe('SC-4 — reconciliation with manual edits', () => {
  test.use({ viewport: DESKTOP });

  test('a removed condition is not re-added while its words stand', async ({ page }) => {
    await mockAccess(page);
    await mockTranslate(page, TRANSLATED);
    await mockScan(page);
    await page.goto(PAGE);
    await typeSentence(page, SENTENCE);
    await expect(page.getByTestId('translated-card')).toHaveCount(2, { timeout: 10_000 });

    // Remove the second chip (« zone jamais testée », from « jamais testé »).
    await page.getByTestId('translated-card').nth(1).getByRole('button', { name: /Retirer/i }).click();
    await expect(page.getByTestId('translated-card')).toHaveCount(1);

    // Keep writing. M.I.A re-proposes it every time; the removal holds.
    await typeSentence(page, ', avec une poche de liquidité prise récemment');
    await page.waitForTimeout(2500);
    await expect(page.getByTestId('translated-card')).toHaveCount(1);
  });

  test('it comes back once the same idea is written in other words', async ({ page }) => {
    await mockAccess(page);
    await mockTranslate(page, TRANSLATED);
    await mockScan(page);
    await page.goto(PAGE);
    await typeSentence(page, SENTENCE);
    await expect(page.getByTestId('translated-card')).toHaveCount(2, { timeout: 10_000 });

    await page.getByTestId('translated-card').nth(1).getByRole('button', { name: /Retirer/i }).click();
    await expect(page.getByTestId('translated-card')).toHaveCount(1);

    // The server now attributes the condition to a NEW fragment: a deliberate
    // re-description, so the removal no longer applies.
    await page.unroute('**/api/scanner/translate');
    await mockTranslate(page, REDESCRIBED);
    await page.getByTestId('describe-input').fill('');
    await typeSentence(page, 'Un Order Block sur une zone vierge en tendance haussière');

    await expect(page.getByTestId('translated-card')).toHaveCount(2, { timeout: 10_000 });
  });
});

test.describe('SC-4 — adversarial gestures', () => {
  test.use({ viewport: DESKTOP });

  test('typing then erasing quickly spends no translation', async ({ page }) => {
    await mockAccess(page);
    const calls = await mockTranslate(page, TRANSLATED);
    await mockScan(page);
    await page.goto(PAGE);

    const input = page.getByTestId('describe-input');
    await input.pressSequentially('Un Order Block jamais', { delay: 5 });
    await input.fill('');
    await page.waitForTimeout(2500);

    expect(calls).toHaveLength(0);
    await expect(page.getByTestId('translated-card')).toHaveCount(0);
    await expect(page.getByTestId('live-status')).toHaveAttribute('data-status', 'idle');
  });

  test('erasing a read sentence clears the reading, not just the field', async ({ page }) => {
    await mockAccess(page);
    await mockTranslate(page, TRANSLATED);
    await mockScan(page);
    await page.goto(PAGE);
    await typeSentence(page, SENTENCE);
    await expect(page.getByTestId('translated-card')).toHaveCount(2, { timeout: 10_000 });

    await page.getByTestId('describe-input').fill('');
    await expect(page.getByTestId('translated-card')).toHaveCount(0);
  });

  test('pasting a long text reads it once, not once per character', async ({ page }) => {
    await mockAccess(page);
    const calls = await mockTranslate(page, TRANSLATED);
    await mockScan(page);
    await page.goto(PAGE);

    // A paste lands as one change event.
    await page.getByTestId('describe-input').fill(
      'Un Order Block jamais testé, en tendance haussière, avec le 1 h qui va dans le même sens et une poche de liquidité prise récemment.',
    );
    await expect(page.getByTestId('translated-card')).toHaveCount(2, { timeout: 10_000 });
    await page.waitForTimeout(1500);

    expect(calls.length).toBeLessThanOrEqual(2);
  });

  test('a refusal renders in place and never takes the field down', async ({ page }) => {
    await mockAccess(page);
    await mockTranslate(page, REFUSED);
    await mockScan(page);
    await page.goto(PAGE);

    const text = 'Montre-moi les meilleurs marchés à trader maintenant';
    await typeSentence(page, text);

    await expect(page.getByTestId('refusal-block')).toBeVisible({ timeout: 10_000 });
    // The field is still there, and still holds what the user wrote.
    await expect(page.getByTestId('describe-input')).toHaveValue(text);
    await expect(page.getByTestId('refusal-example').first()).toBeVisible();
    await noRawKeys(page);
  });
});
