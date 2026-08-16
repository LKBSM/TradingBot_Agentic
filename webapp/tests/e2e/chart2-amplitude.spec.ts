import { expect, test, type Page, type TestInfo } from '@playwright/test';
import { FIXTURE_XAU_M15 } from '../../lib/market-reading/fixtures';
import { dismissCookieBanner } from './utils';

/**
 * CHART-2 — amplitude, zoom controls (Variante 1), honest edge notices.
 *
 * Runs at three viewports via the two Playwright projects: 1280×800 and 1920×1080
 * under chromium-desktop (hover: hover), 390×844 under the iPhone-12 project
 * (hover: none, touch). Each `describe` self-selects the project whose pointer
 * type matches, so the hover-reveal vs always-visible control behaviour is tested
 * on the pointer type it actually targets.
 *
 * Asserts:
 *   · default view renders + the discreet control toolbar exists;
 *   · controls appear on hover (desktop) / stay visible (mobile), and are
 *     keyboard-reachable (focus-within reveals them);
 *   · dezooming loads older history on demand (a `before=` request) WITHOUT the
 *     candles disappearing, then honestly says « début des données » at the real
 *     limit and « hors fenêtre d'analyse » past the analysed window;
 *   · a failed history page surfaces a retry affordance, never a silent void.
 */

const TOTAL = 900;
const START = Math.floor(Date.UTC(2026, 0, 1) / 1000);
const STEP = 900; // M15 = 15 min
const ANALYSIS_WINDOW_BARS = 300;

function candleAt(i: number) {
  const close = 2000 + i;
  return { time: START + i * STEP, open: close - 0.5, high: close + 1, low: close - 1, close, volume: 100 };
}
const ALL = Array.from({ length: TOTAL }, (_, i) => candleAt(i));

/** Mirror of the backend: most-recent window, or a strictly-older page on `before`. */
function candleBody(url: string) {
  const u = new URL(url);
  const limit = Number(u.searchParams.get('limit') ?? '200');
  const before = u.searchParams.get('before');
  let slice: ReturnType<typeof candleAt>[];
  if (before !== null) {
    const b = Number(before);
    const older = ALL.filter((c) => c.time < b);
    slice = older.slice(Math.max(0, older.length - limit));
  } else {
    slice = ALL.slice(Math.max(0, ALL.length - limit));
  }
  const oldestReturned = slice.length ? slice[0]!.time : null;
  const has_more_history = oldestReturned !== null && oldestReturned > ALL[0]!.time;
  return { instrument: 'XAUUSD', timeframe: 'M15', candles: slice, has_more_history };
}

const READING = {
  ...FIXTURE_XAU_M15,
  header: { ...FIXTURE_XAU_M15.header, analysis_window_bars: ANALYSIS_WINDOW_BARS },
};

async function gotoApp(page: Page, w: number, h: number) {
  await page.setViewportSize({ width: w, height: h });
  await page.goto('/app?instrument=XAUUSD&timeframe=M15');
  await dismissCookieBanner(page);
  if (w < 1280) {
    const lecture = page.getByRole('tab', { name: /Lecture/i });
    await lecture.waitFor({ state: 'visible', timeout: 8_000 });
    await lecture.click();
  }
  await expect(page.locator('canvas').first()).toBeVisible({ timeout: 15_000 });
}

const isMobileProject = (info: TestInfo) => info.project.name.includes('mobile');

/** Hover the plot via raw mouse move (locator.hover() times out on the stacked
 *  lightweight-charts canvases — they "intercept pointer events"). */
async function hoverChart(page: Page) {
  const box = await page.locator('canvas').first().boundingBox();
  if (!box) throw new Error('no chart canvas');
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
}

/** Drive the chart to the oldest edge from the keyboard (deterministic on every
 *  viewport — no canvas wheel/hover). Dezoom a little, then pan left repeatedly:
 *  each pan crosses the load threshold (fetching older pages) until the true start.
 *  On a narrow plot the spacing floor caps visible bars, so panning — not just
 *  zooming — is what reaches the left edge. */
async function panToStart(page: Page) {
  const region = page.getByRole('application', { name: /Graphique/i });
  await region.focus();
  for (let i = 0; i < 4; i += 1) await page.keyboard.press('-');
  for (let i = 0; i < 60; i += 1) {
    await page.keyboard.press('ArrowLeft');
    await page.waitForTimeout(60);
  }
}

interface VP {
  name: string;
  w: number;
  h: number;
  touch: boolean;
}
const VIEWPORTS: VP[] = [
  { name: '1280x800', w: 1280, h: 800, touch: false },
  { name: '1920x1080', w: 1920, h: 1080, touch: false },
  { name: '390x844', w: 390, h: 844, touch: true },
];

for (const vp of VIEWPORTS) {
  test.describe(`CHART-2 — ${vp.name}`, () => {
    let beforeRequests = 0;
    let failOlder = false;

    test.beforeEach(async ({ page }) => {
      // A touch viewport only makes sense under the mobile (touch) project, and a
      // pointer viewport under the desktop project — so the hover media matches.
      test.skip(vp.touch !== isMobileProject(test.info()), 'viewport/project pointer mismatch');
      beforeRequests = 0;
      failOlder = false;
      await page.route('**/api/candles**', (route) => {
        const url = route.request().url();
        if (url.includes('before=')) {
          beforeRequests += 1;
          if (failOlder) return route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'down' }) });
        }
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(candleBody(url)) });
      });
      await page.route('**/api/market-reading**', (route) =>
        route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(READING) }),
      );
      await page.route('**/api/market-status**', (route) =>
        route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) }),
      );
    });

    const toolbar = (page: Page) => page.getByRole('group', { name: /Contrôles du graphique/i });
    const zoomOutBtn = (page: Page) => page.getByRole('button', { name: /Zoom arrière/i });

    test('default view renders with the discreet control toolbar', async ({ page }) => {
      await gotoApp(page, vp.w, vp.h);
      await expect(toolbar(page)).toBeAttached();
      // The four core controls exist (zoom in/out, recent, default view).
      await expect(page.getByRole('button', { name: /Zoom avant/i })).toBeAttached();
      await expect(zoomOutBtn(page)).toBeAttached();
      await expect(page.getByRole('button', { name: /bougie la plus récente/i })).toBeAttached();
      await expect(page.getByRole('button', { name: /vue par défaut/i })).toBeAttached();
      // No premature edge notice on the opening (recent) view.
      await expect(page.getByText(/Début des données disponibles/i)).toHaveCount(0);
    });

    test('controls reveal on hover (desktop) / stay visible (mobile) + keyboard-reachable', async ({ page }, info) => {
      await gotoApp(page, vp.w, vp.h);
      const bar = toolbar(page);
      if (isMobileProject(info)) {
        // Coarse pointer: controls stay visible (pinch isn't enough for everyone).
        await expect(bar).toHaveCSS('opacity', '1');
      } else {
        // Fine pointer: hidden at rest, revealed on hover.
        await expect(bar).toHaveCSS('opacity', '0');
        await hoverChart(page);
        await expect(bar).toHaveCSS('opacity', '1');
        // And reachable by keyboard even without hover: focus-within reveals it.
        await page.mouse.move(0, 0);
        await expect(bar).toHaveCSS('opacity', '0');
        await page.getByRole('button', { name: /Zoom avant/i }).focus();
        await expect(bar).toHaveCSS('opacity', '1');
      }
    });

    test('the chart region is a focusable keyboard target', async ({ page }) => {
      await gotoApp(page, vp.w, vp.h);
      const region = page.getByRole('application', { name: /Graphique/i });
      await region.focus();
      await expect(region).toBeFocused();
      // Keyboard zoom/pan verbs don't throw and keep focus on the region.
      await page.keyboard.press('-');
      await page.keyboard.press('Minus');
      await page.keyboard.press('ArrowLeft');
      await expect(region).toBeFocused();
    });

    test('dezoom loads older history, keeps candles, then says « début des données » + « hors fenêtre d\'analyse »', async ({ page }) => {
      await gotoApp(page, vp.w, vp.h);
      await panToStart(page);
      // A backward page was fetched on demand (never the whole depth at once).
      expect(beforeRequests).toBeGreaterThan(0);
      // The candles never vanished during the load — the canvas stayed present.
      await expect(page.locator('canvas').first()).toBeVisible();
      // Honest limit notice + out-of-analysis-window notice.
      await expect(page.getByText(/Début des données disponibles/i)).toBeVisible({ timeout: 10_000 });
      await expect(page.getByText(/Hors de la fenêtre d'analyse/i)).toBeVisible();
    });

    test('a failed history page surfaces a retry affordance, never a silent void', async ({ page }) => {
      await gotoApp(page, vp.w, vp.h);
      failOlder = true;
      await panToStart(page);
      // Scope to the chart region — other panels on the desktop page also carry a
      // "Réessayer" (calendar/reading), which would otherwise clash in strict mode.
      const region = page.getByRole('application', { name: /Graphique/i });
      await expect(region.getByText(/Historique indisponible/i)).toBeVisible({ timeout: 10_000 });
      await expect(region.getByRole('button', { name: /Réessayer/i })).toBeVisible();
      // Candles still on screen despite the failed page.
      await expect(page.locator('canvas').first()).toBeVisible();
    });
  });
}
