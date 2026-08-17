import { expect, test, type Page } from '@playwright/test';
import { CONDITION_PALETTE } from '@/lib/conditions/palette';

/**
 * UI-2 — enforceable invariants for the scanner's typographic tenue. Runs the two
 * tab entry points (Décrire + Conditions) at both viewports in fr + en and asserts:
 *   1. at most SIX distinct font sizes are rendered under `.app-shell`;
 *   2. every rendered size belongs to the six-step scale (no stray 16px base);
 *   3. the monospace face is on VALUES only — the language notes are proportional,
 *      while a timeframe CODE stays monospace (a sanctioned value use);
 *   4. the condition count shown is DERIVED (equals the real palette length);
 *   5. no raw i18n key leaks (no fr/en fallback hole);
 *   6. the describe content fits the first fold at 1280×800.
 */

const ALLOWED = new Set(['26px', '19px', '15px', '14px', '12px', '11px']);
const COUNT = CONDITION_PALETTE.length; // derived, mirrors the backend palette

async function mock(page: Page) {
  await page.route('**/api/access/me', (r) =>
    r.fulfill({
      json: {
        authenticated: true, gate_enforced: false, beta_lockdown: false,
        must_login: false, is_owner: true, has_access: true, subscription_required: false,
      },
    }),
  );
  await page.addInitScript(() => {
    class Fake {
      onstart: (() => void) | null = null;
      onend: (() => void) | null = null;
      start() { this.onstart?.(); }
      stop() { this.onend?.(); }
      abort() {}
    }
    (window as unknown as Record<string, unknown>).SpeechRecognition = Fake;
    (window as unknown as Record<string, unknown>).webkitSpeechRecognition = Fake;
  });
}

/** Every visible, text-bearing element under `.app-shell`, with its font metrics. */
async function textNodes(page: Page) {
  return page.evaluate(() => {
    const out: Array<{ size: string; mono: boolean; text: string }> = [];
    const root = document.querySelector('.app-shell');
    if (!root) return out;
    for (const el of Array.from(root.querySelectorAll<HTMLElement>('*'))) {
      const direct = Array.from(el.childNodes).some(
        (n) => n.nodeType === 3 && (n.textContent ?? '').trim().length > 0,
      );
      if (!direct) continue;
      const r = el.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) continue;
      const cs = getComputedStyle(el);
      if (cs.visibility === 'hidden' || cs.display === 'none') continue;
      out.push({
        size: cs.fontSize,
        mono: /mono|jetbrains/i.test(cs.fontFamily),
        text: (el.textContent ?? '').trim().slice(0, 40),
      });
    }
    return out;
  });
}

const VIEWPORTS = [
  { tag: 'desktop', width: 1280, height: 800 },
  { tag: 'mobile', width: 390, height: 844 },
] as const;

for (const loc of ['fr', 'en'] as const) {
  for (const vp of VIEWPORTS) {
    test(`scale + mono discipline — ${loc} ${vp.tag}`, async ({ page }) => {
      await mock(page);
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await page.goto(`/${loc}/scanner/decrire`);
      await page.getByTestId('describe-input').waitFor({ state: 'visible' });

      const nodes = await textNodes(page);

      // (1)+(2) — at most six sizes, all on the scale.
      const sizes = [...new Set(nodes.map((n) => n.size))];
      const offScale = sizes.filter((s) => !ALLOWED.has(s));
      expect(offScale, `off-scale sizes: ${offScale.join(', ')} — samples: ${
        nodes.filter((n) => offScale.includes(n.size)).map((n) => `${n.size}:"${n.text}"`).join(' | ')
      }`).toEqual([]);
      expect(sizes.length, `distinct sizes: ${sizes.join(', ')}`).toBeLessThanOrEqual(6);

      // (3) — the language notes are NOT monospace…
      for (const id of ['mia-sub', 'scope-note', 'transcription-note', 'examples-label']) {
        const fam = await page.getByTestId(id).evaluate((el) => getComputedStyle(el).fontFamily);
        expect(fam, `${id} must be proportional`).not.toMatch(/mono|jetbrains/i);
      }
      // …while a timeframe CODE (a value) stays monospace.
      const tf = page.locator('.app-shell .tf').first();
      if (await tf.count()) {
        const tfFam = await tf.evaluate((el) => getComputedStyle(el).fontFamily);
        expect(tfFam, 'timeframe code should stay monospace (value)').toMatch(/mono|jetbrains/i);
      }

      // (5) — no raw i18n key on screen.
      expect(await page.locator('body').innerText()).not.toMatch(/scannerChat\.[a-zA-Z0-9_.]+/);
    });
  }
}

test('condition count is derived from the palette (fr)', async ({ page }) => {
  await mock(page);
  await page.goto('/fr/scanner/decrire');
  await page.getByTestId('scope-note').waitFor({ state: 'visible' });
  await expect(page.getByTestId('scope-note')).toContainText(String(COUNT));
});

test('describe content fits the first fold at 1280×800 (fr)', async ({ page }) => {
  await mock(page);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/fr/scanner/decrire');
  await page.getByTestId('example-chip').last().waitFor({ state: 'visible' });
  // The last example card's bottom must sit within the first viewport height.
  const bottom = await page
    .getByTestId('example-chip')
    .last()
    .evaluate((el) => el.getBoundingClientRect().bottom);
  expect(bottom).toBeLessThanOrEqual(800);
});

test('the manual Conditions tab is also on-scale (fr desktop)', async ({ page }) => {
  await mock(page);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/fr/scanner');
  await page.locator('.app-shell').waitFor({ state: 'visible' });
  await page.waitForTimeout(500);
  const nodes = await textNodes(page);
  const sizes = [...new Set(nodes.map((n) => n.size))];
  const offScale = sizes.filter((s) => !ALLOWED.has(s));
  expect(offScale, `off-scale: ${offScale.join(', ')} — ${
    nodes.filter((n) => offScale.includes(n.size)).map((n) => `${n.size}:"${n.text}"`).join(' | ')
  }`).toEqual([]);
  expect(sizes.length).toBeLessThanOrEqual(6);
});
