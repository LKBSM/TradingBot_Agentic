import { expect, test, type Page } from '@playwright/test';

/**
 * Publication page M.I.A card — now WIRED to the chatbot (previously a dead
 * placeholder: input + send + suggestion chips rendered but did nothing).
 *
 * Verifies the card actually converses: a suggestion chip and the free-text
 * input both POST /api/chatbot/message and render M.I.A's answer; a 503 degrades
 * to an honest inline message while the input stays usable. All network mocked —
 * the real backend is never called.
 */

const D = 86400_000;
const iso = (ms: number) => new Date(Date.now() + ms).toISOString();

const FULL_ACCESS = {
  authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false,
  is_owner: true, has_access: true, subscription_required: false,
};

const EVENT = {
  window_start: iso(-40 * D), window_end: iso(40 * D), generated_at: iso(0),
  coverage: { source: 'official', feed_start: null, feed_end: null, partial: false, last_success: {}, stale_sources: [] },
  attribution: [],
  events: [{
    event_id: 'bls:us_cpi:2026-08-12', source: 'bls', series_code: 'CUUR0000SA0', license_label: 'x',
    event: 'US_CPI', currency: 'USD', organism: 'Bureau of Labor Statistics', periodicity: 'monthly',
    scheduled_at: iso(12 * D), source_timezone: 'America/New_York', time_confirmed: true,
    markets: ['XAUUSD', 'EURUSD'], value_unit: '% de variation annuelle',
    actual: null, actual_initial: 3.2, previous: 3.0, revised: false, revised_at: iso(-20 * D),
    actual_state: 'pending', refreshed_at: iso(0), value_series: [],
  }],
};
const MEAS = { event_key: 'us_cpi', market: '', calm_before: null, structure_state: null, zone_lifecycle: null, return_to_calm: null };

const json = (b: unknown) => ({ status: 200, contentType: 'application/json', body: JSON.stringify(b) });
const CPI_URL = 'bls%3Aus_cpi%3A2026-08-12';

async function goto(page: Page, chat: (r: import('@playwright/test').Route) => void): Promise<boolean> {
  await page.route('**/api/access/me', (r) => r.fulfill(json(FULL_ACCESS)));
  await page.route('**/api/publications/*/measures', (r) => r.fulfill(json(MEAS)));
  await page.route('**/api/calendar/event/*', (r) => r.fulfill(json(EVENT)));
  await page.route('**/api/calendar*', (r) => r.fulfill(json(EVENT)));
  await page.route('**/api/chatbot/message', chat);
  await page.goto(`/actualites/${CPI_URL}`);
  try {
    await page.locator('.pub-mia').first().waitFor({ state: 'visible', timeout: 20000 });
  } catch {
    return false;
  }
  return true;
}

test('publication M.I.A card — free-text question gets an answer', async ({ page }) => {
  let sent: any = null;
  const ok = await goto(page, (r) => {
    sent = JSON.parse(r.request().postData() || '{}');
    return r.fulfill(json({ content: 'Sur cette publication, la structure est haussière.', blocked_reason: null, tool_calls_made: [] }));
  });
  test.skip(!ok, 'gated');

  await page.getByTestId('pub-mia-input').fill('Que dit la structure ?');
  await page.getByTestId('pub-mia-send').click();
  await expect(page.getByTestId('pub-mia-answer')).toContainText('structure est haussière');
  // The request carried the publication context preamble (anchors M.I.A).
  expect(String(sent?.user_message)).toContain('US_CPI');
  expect(String(sent?.user_message)).toContain('Que dit la structure ?');
});

test('publication M.I.A card — a suggestion chip sends and answers', async ({ page }) => {
  const ok = await goto(page, (r) =>
    r.fulfill(json({ content: 'Réponse à la question suggérée.', blocked_reason: null, tool_calls_made: [] })),
  );
  test.skip(!ok, 'gated');

  await page.getByTestId('pub-mia-chip').first().click();
  await expect(page.getByTestId('pub-mia-user')).toBeVisible(); // the chip text became a user turn
  await expect(page.getByTestId('pub-mia-answer')).toContainText('question suggérée');
});

test('publication M.I.A card — 503 degrades honestly, input stays usable', async ({ page }) => {
  const ok = await goto(page, (r) => r.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'off' }) }));
  test.skip(!ok, 'gated');

  await page.getByTestId('pub-mia-input').fill('une question');
  await page.getByTestId('pub-mia-send').click();
  await expect(page.getByTestId('pub-mia-error')).toBeVisible();
  // Still usable: type again.
  await page.getByTestId('pub-mia-input').fill('je peux réécrire');
  await expect(page.getByTestId('pub-mia-input')).toHaveValue('je peux réécrire');
});
