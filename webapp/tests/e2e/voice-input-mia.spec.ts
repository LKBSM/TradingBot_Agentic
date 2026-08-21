import { expect, test, type Page } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * Voice dictation extended to every M.I.A chat surface (mission "Dictée vocale").
 *
 * The scanner "Décris ta stratégie" is already covered at both viewports by
 * sc2-scanner-conversationnel.spec.ts. This spec proves the SAME shared browser
 * dictation now works on the three OTHER surfaces — the docked /app M.I.A Agent,
 * the /zones panel and the /actualites publication chat — with the three honest
 * states (supported / permission denied / unsupported) and, crucially, that the
 * transcribed text is EXACTLY what fills the field and what is sent to M.I.A
 * (no hidden transform).
 *
 * All network is mocked; the real backend is never called. A scripted
 * SpeechRecognition is injected before app code runs.
 */

const FULL_ACCESS = {
  authenticated: true, gate_enforced: false, beta_lockdown: false, must_login: false,
  is_owner: true, has_access: true, subscription_required: false,
};

const D = 86400_000;
const iso = (ms: number) => new Date(Date.now() + ms).toISOString();

// ── /app + /zones reading mocks ────────────────────────────────────────────
const START = Math.floor(Date.UTC(2026, 0, 1) / 1000);
const CANDLES = Array.from({ length: 60 }, (_, i) => {
  const close = 2000 + i;
  return { time: START + i * 900, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
});

async function mockProduct(page: Page) {
  await page.route('**/api/access/me', (r) => r.fulfill({ json: FULL_ACCESS }));
  await page.route('**/api/candles**', (r) =>
    r.fulfill({ json: { instrument: 'XAUUSD', timeframe: 'M15', candles: CANDLES, has_more_history: false } }),
  );
  await page.route('**/api/market-reading**', (r) => r.fulfill({ json: FIXTURE_XAU_M15 }));
  await page.route('**/api/market-status**', (r) => r.fulfill({ json: {} }));
}

// ── /actualites publication mock (mirrors pub-mia-chat.spec) ────────────────
const CPI_URL = 'bls%3Aus_cpi%3A2026-08-12';
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

async function mockPublication(page: Page, chat?: (r: import('@playwright/test').Route) => void) {
  await page.route('**/api/access/me', (r) => r.fulfill({ json: FULL_ACCESS }));
  await page.route('**/api/publications/*/measures', (r) => r.fulfill({ json: MEAS }));
  await page.route('**/api/calendar/event/*', (r) => r.fulfill({ json: EVENT }));
  await page.route('**/api/calendar*', (r) => r.fulfill({ json: EVENT }));
  if (chat) await page.route('**/api/chatbot/message', chat);
}

// ── Scripted browser dictation ─────────────────────────────────────────────
/** Inject a fake SpeechRecognition BEFORE app code; the latest instance is kept
 *  on window so a test can drive a final transcript through it. */
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
      constructor() {
        (window as unknown as Record<string, unknown>).__lastRec = this;
      }
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

async function removeSpeech(page: Page) {
  await page.addInitScript(() => {
    delete (window as unknown as Record<string, unknown>).SpeechRecognition;
    delete (window as unknown as Record<string, unknown>).webkitSpeechRecognition;
  });
}

/** Fire a finalised transcript through the most recent recognition instance. */
async function speak(page: Page, transcript: string) {
  await page.evaluate((text: string) => {
    const rec = (window as unknown as { __lastRec?: { onresult?: (e: unknown) => void } }).__lastRec;
    rec?.onresult?.({
      resultIndex: 0,
      results: { length: 1, 0: { isFinal: true, length: 1, 0: { transcript: text } } },
    });
  }, transcript);
}

// ════════════════════════════════════════════════════════════════════════════
// Desktop 1280×800 — the three surfaces, three states each.
// ════════════════════════════════════════════════════════════════════════════
test.describe('Voice dictation — desktop 1280×800', () => {
  test.use({ viewport: { width: 1280, height: 800 } });
  test.setTimeout(90_000);

  // ── /app docked M.I.A Agent ───────────────────────────────────────────────
  test('/app chat — mic dictates exactly into the field', async ({ page }) => {
    await mockProduct(page);
    await installFakeSpeech(page, 'ok');
    await page.goto('/app?instrument=XAUUSD&timeframe=M15');
    await dismissCookieBanner(page);

    const form = page.locator('form:has([data-testid="chat-input"])').first();
    const mic = form.getByTestId('mic-button');
    const field = page.getByTestId('chat-input').first();
    await expect(mic).toBeVisible();

    await mic.click();
    await speak(page, 'quelles zones sont actives');
    // The transcript is EXACTLY what fills the field — no transform.
    await expect(field).toHaveValue('quelles zones sont actives');
  });

  test('/app chat — permission denied shows a message, field stays usable', async ({ page }) => {
    await mockProduct(page);
    await installFakeSpeech(page, 'denied');
    await page.goto('/app?instrument=XAUUSD&timeframe=M15');
    await dismissCookieBanner(page);

    const form = page.locator('form:has([data-testid="chat-input"])').first();
    await form.getByTestId('mic-button').click();
    await expect(page.getByTestId('dictation-error')).toBeVisible();
    // Keyboard still works.
    await page.getByTestId('chat-input').first().fill('je tape à la place');
    await expect(page.getByTestId('chat-input').first()).toHaveValue('je tape à la place');
  });

  test('/app chat — unsupported browser hides the mic (no dead button)', async ({ page }) => {
    await mockProduct(page);
    await removeSpeech(page);
    await page.goto('/app?instrument=XAUUSD&timeframe=M15');
    await dismissCookieBanner(page);

    const form = page.locator('form:has([data-testid="chat-input"])').first();
    await expect(page.getByTestId('chat-input').first()).toBeVisible();
    await expect(form.getByTestId('mic-button')).toHaveCount(0);
  });

  // ── /zones M.I.A panel ────────────────────────────────────────────────────
  async function openZonePanel(page: Page) {
    await page.goto('/zones?instrument=XAUUSD&timeframe=M15', { waitUntil: 'domcontentloaded' });
    await dismissCookieBanner(page);
    const card = page.locator('[data-zone-id]').first();
    await card.waitFor({ state: 'visible', timeout: 60_000 });
    await card.click();
    await page.locator('.zmia-input:visible').waitFor({ state: 'visible', timeout: 20_000 });
  }

  test('/zones panel — mic dictates exactly, submit carries the transcript', async ({ page }) => {
    await mockProduct(page);
    await installFakeSpeech(page, 'ok');
    await openZonePanel(page);

    const panel = page.locator('.zmia:visible');
    const mic = page.locator('.zmia-input:visible').getByTestId('mic-button');
    const field = page.locator('.zmia-input:visible input');
    await expect(mic).toBeVisible();

    await mic.click();
    await speak(page, 'quelle est la proximité');
    await expect(field).toHaveValue('quelle est la proximité');

    // Submitting routes the exact transcript through as a user turn (0 network).
    await field.press('Enter');
    await expect(panel.locator('.bub.u').last()).toHaveText('quelle est la proximité');
  });

  test('/zones panel — permission denied shows a message, field stays usable', async ({ page }) => {
    await mockProduct(page);
    await installFakeSpeech(page, 'denied');
    await openZonePanel(page);

    const panel = page.locator('.zmia:visible');
    await page.locator('.zmia-input:visible').getByTestId('mic-button').click();
    await expect(panel.getByTestId('dictation-error')).toBeVisible();
    await page.locator('.zmia-input:visible input').fill('je tape');
    await expect(page.locator('.zmia-input:visible input')).toHaveValue('je tape');
  });

  test('/zones panel — unsupported browser hides the mic', async ({ page }) => {
    await mockProduct(page);
    await removeSpeech(page);
    await openZonePanel(page);
    await expect(page.locator('.zmia-input:visible').getByTestId('mic-button')).toHaveCount(0);
    await expect(page.locator('.zmia-input:visible input')).toBeVisible();
  });

  // ── /actualites publication M.I.A chat ────────────────────────────────────
  test('/actualites chat — dictation reaches the backend verbatim', async ({ page }) => {
    let sent: any = null;
    await mockPublication(page, (r) => {
      sent = JSON.parse(r.request().postData() || '{}');
      return r.fulfill({ json: { content: 'Réponse.', blocked_reason: null, tool_calls_made: [] } });
    });
    await installFakeSpeech(page, 'ok');
    await page.goto(`/actualites/${CPI_URL}`);
    await dismissCookieBanner(page);
    const form = page.locator('.pub-mia-form');
    await form.waitFor({ state: 'visible', timeout: 20_000 });

    await form.getByTestId('mic-button').click();
    await speak(page, 'que dit la structure');
    await expect(page.getByTestId('pub-mia-input')).toHaveValue('que dit la structure');

    await page.getByTestId('pub-mia-send').click();
    await expect(page.getByTestId('pub-mia-answer')).toContainText('Réponse.');
    // What M.I.A received is exactly the dictated text (plus the publication anchor).
    expect(String(sent?.user_message)).toContain('que dit la structure');
  });

  test('/actualites chat — permission denied shows a message, field stays usable', async ({ page }) => {
    await mockPublication(page);
    await installFakeSpeech(page, 'denied');
    await page.goto(`/actualites/${CPI_URL}`);
    await dismissCookieBanner(page);
    const form = page.locator('.pub-mia-form');
    await form.waitFor({ state: 'visible', timeout: 20_000 });

    await form.getByTestId('mic-button').click();
    await expect(page.locator('.pub-mia').getByTestId('dictation-error')).toBeVisible();
    await page.getByTestId('pub-mia-input').fill('je tape');
    await expect(page.getByTestId('pub-mia-input')).toHaveValue('je tape');
  });

  test('/actualites chat — unsupported browser hides the mic', async ({ page }) => {
    await mockPublication(page);
    await removeSpeech(page);
    await page.goto(`/actualites/${CPI_URL}`);
    await dismissCookieBanner(page);
    const form = page.locator('.pub-mia-form');
    await form.waitFor({ state: 'visible', timeout: 20_000 });
    await expect(form.getByTestId('mic-button')).toHaveCount(0);
    await expect(page.getByTestId('pub-mia-input')).toBeVisible();
  });
});

// ════════════════════════════════════════════════════════════════════════════
// Mobile 390×844 — the publication chat (a normal card, present on phones).
// The scanner surface is covered at 390×844 by sc2-scanner-conversationnel.
// ════════════════════════════════════════════════════════════════════════════
test.describe('Voice dictation — mobile 390×844', () => {
  test.use({ viewport: { width: 390, height: 844 } });
  test.setTimeout(90_000);

  test('/actualites chat — mic present and dictates exactly on phone', async ({ page }) => {
    await mockPublication(page, (r) =>
      r.fulfill({ json: { content: 'Réponse.', blocked_reason: null, tool_calls_made: [] } }),
    );
    await installFakeSpeech(page, 'ok');
    await page.goto(`/actualites/${CPI_URL}`);
    await dismissCookieBanner(page);
    const form = page.locator('.pub-mia-form');
    await form.waitFor({ state: 'visible', timeout: 20_000 });

    await expect(form.getByTestId('mic-button')).toBeVisible();
    await form.getByTestId('mic-button').click();
    await speak(page, 'quel est le consensus');
    await expect(page.getByTestId('pub-mia-input')).toHaveValue('quel est le consensus');
  });

  test('/actualites chat — unsupported browser hides the mic on phone', async ({ page }) => {
    await mockPublication(page);
    await removeSpeech(page);
    await page.goto(`/actualites/${CPI_URL}`);
    await dismissCookieBanner(page);
    const form = page.locator('.pub-mia-form');
    await form.waitFor({ state: 'visible', timeout: 20_000 });
    await expect(form.getByTestId('mic-button')).toHaveCount(0);
  });
});
