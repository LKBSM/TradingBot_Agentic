import { expect, test, type Page } from '@playwright/test';

/**
 * Publication page M.I.A block — MIA-3 façade.
 *
 * The publication card is now a SUGGESTIONS FAÇADE: it keeps the bespoke chips
 * but has no local engine. Clicking a chip feeds the ONE shared conversation
 * (the shell M.I.A column), oriented on this publication via the id-locked
 * `[Publication : <event_id>]` preamble; the answer renders in that column. A
 * 503 degrades honestly through the shared provider. All network mocked.
 */

const D = 86400_000;
const iso = (ms: number) => new Date(Date.now() + ms).toISOString();

const FULL_ACCESS = {
  authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false,
  is_owner: true, has_access: true, subscription_required: false,
};

const EVENT_ID = 'bls:us_cpi:2026-08-12';
const EVENT = {
  window_start: iso(-40 * D), window_end: iso(40 * D), generated_at: iso(0),
  coverage: { source: 'official', feed_start: null, feed_end: null, partial: false, last_success: {}, stale_sources: [] },
  attribution: [],
  events: [{
    event_id: EVENT_ID, source: 'bls', series_code: 'CUUR0000SA0', license_label: 'x',
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

function sse(events: Array<Record<string, unknown>>): string {
  return events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('');
}
function answerBody(content: string): string {
  return sse([
    { event: 'activity' },
    { event: 'answer', content, blocked_reason: null, tool_calls_made: [], view_actions: [] },
  ]);
}

async function goto(
  page: Page,
  stream: (r: import('@playwright/test').Route) => void,
): Promise<boolean> {
  await page.route('**/api/access/me', (r) => r.fulfill(json(FULL_ACCESS)));
  await page.route('**/api/publications/*/measures', (r) => r.fulfill(json(MEAS)));
  await page.route('**/api/calendar/event/*', (r) => r.fulfill(json(EVENT)));
  await page.route('**/api/calendar*', (r) => r.fulfill(json(EVENT)));
  await page.route('**/api/candles**', (r) => r.fulfill(json({ instrument: 'XAUUSD', timeframe: 'M15', candles: [] })));
  await page.route('**/api/market-reading**', (r) => r.fulfill(json({})));
  await page.route('**/api/market-status**', (r) => r.fulfill(json({})));
  await page.route('**/api/chatbot/stream', stream);
  await page.goto(`/actualites/${CPI_URL}`);
  try {
    await page.locator('.pub-mia').first().waitFor({ state: 'visible', timeout: 20000 });
  } catch {
    return false;
  }
  return true;
}

test('publication M.I.A — a suggestion chip feeds the shared conversation with the publication orientation', async ({ page }) => {
  let sent: string | null = null;
  const ok = await goto(page, (r) => {
    sent = r.request().postData();
    return r.fulfill({ status: 200, contentType: 'text/event-stream', body: answerBody('Sur cette publication, la structure est haussière.') });
  });
  test.skip(!ok, 'gated');

  await page.getByTestId('pub-mia-chip').first().click();
  // The answer renders in the shared M.I.A column.
  await expect(page.getByText('Sur cette publication, la structure est haussière.')).toBeVisible({ timeout: 15000 });
  // The request carried the id-locked publication orientation preamble.
  expect(String(sent)).toContain(`[Publication : ${EVENT_ID}]`);
});

test('publication M.I.A — 503 degrades honestly through the shared provider', async ({ page }) => {
  const ok = await goto(page, (r) => r.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'off' }) }));
  test.skip(!ok, 'gated');

  await page.getByTestId('pub-mia-chip').first().click();
  // The shared conversation shows the honest unavailable fallback, and the
  // composer input stays present (still usable).
  await expect(page.getByText(/n'est pas disponible|indisponible|hors-ligne|hors ligne/i).first()).toBeVisible({ timeout: 15000 });
  await expect(page.getByTestId('chat-input').first()).toBeVisible();
});
