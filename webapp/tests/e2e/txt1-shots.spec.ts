import { test, type Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

/**
 * TXT-1 before/after captures. Run twice with SHOT_DIR=before (on the pre-TXT-1
 * tree) and SHOT_DIR=after (on this branch). Not an assertion spec — it only
 * writes PNGs the founder reviews live before merge.
 */

const DIR = process.env.SHOT_DIR ?? 'after';
const OUT = path.join('..', 'docs', 'audits', 'txt1-shots', DIR);

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
    class Fake { onstart:(()=>void)|null=null; onend:(()=>void)|null=null; start(){this.onstart?.();} stop(){this.onend?.();} abort(){} }
    (window as unknown as Record<string, unknown>).SpeechRecognition = Fake;
    (window as unknown as Record<string, unknown>).webkitSpeechRecognition = Fake;
  });
}

const SCREENS = [
  { name: 'scanner-decrire', url: 'scanner/decrire', ready: 'describe-input' },
  { name: 'scanner', url: 'scanner', ready: null },
  { name: 'zones', url: 'zones', ready: null },
] as const;

for (const loc of ['fr', 'en'] as const) {
  for (const s of SCREENS) {
    test(`shot ${DIR} ${loc} ${s.name}`, async ({ page }) => {
      fs.mkdirSync(OUT, { recursive: true });
      await mock(page);
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto(`/${loc}/${s.url}`, { waitUntil: 'networkidle' });
      if (s.ready) await page.getByTestId(s.ready).waitFor({ state: 'visible' });
      await page.waitForTimeout(1200);
      await page.screenshot({ path: path.join(OUT, `${loc}-${s.name}.png`) });
    });
  }
}
