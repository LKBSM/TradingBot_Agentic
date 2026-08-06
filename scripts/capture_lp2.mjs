// LP-2 diagnostic captures of the REAL product (read-only).
// Runs against frontend on :3400 (proxying /api to backend :8000).
// Output → docs/design/captures/
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const BASE = 'http://localhost:3400';
const OUT = 'C:/MyPythonProjects/wt-lp-2/docs/design/captures';
mkdirSync(OUT, { recursive: true });

const log = (...a) => console.log('[cap]', ...a);

async function dismiss(page) {
  for (const t of ['Accepter', 'Accept', "J'accepte", 'Tout accepter', 'OK']) {
    try {
      const b = page.getByRole('button', { name: t });
      if (await b.count()) { await b.first().click({ timeout: 1000 }); await page.waitForTimeout(300); break; }
    } catch {}
  }
}

async function settle(page, ms = 4500) {
  try { await page.waitForLoadState('networkidle', { timeout: 25000 }); } catch {}
  await page.waitForTimeout(ms);
}

async function shot(page, name, { full = true } = {}) {
  const path = `${OUT}/${name}.png`;
  try { await page.screenshot({ path, fullPage: full }); log('OK', name); }
  catch (e) { log('FAIL', name, e.message); }
}

async function run() {
  const browser = await chromium.launch();

  // ---------- DESKTOP 1280x800 ----------
  const desk = await browser.newContext({ viewport: { width: 1280, height: 800 }, locale: 'fr-FR' });
  const p = await desk.newPage();

  // /app XAUUSD M15
  await p.goto(`${BASE}/fr/app?instrument=XAUUSD&timeframe=M15`, { waitUntil: 'domcontentloaded' });
  await dismiss(p);
  await settle(p, 6000);
  await shot(p, 'desktop-app-xauusd-m15-full');
  await p.evaluate(() => window.scrollTo(0, 0)); await p.waitForTimeout(400);
  await shot(p, 'desktop-app-xauusd-m15-viewport', { full: false });

  // Toggle a layer OFF (before/after) — try FVG then Liquidité
  for (const label of ['FVG', 'Liquidité', 'BOS/CHOCH', 'Mitigées']) {
    try {
      const el = p.getByText(label, { exact: true }).first();
      if (await el.count()) {
        await el.scrollIntoViewIfNeeded();
        await shot(p, `desktop-app-layers-before`, { full: false });
        await el.click({ timeout: 1500 });
        await p.waitForTimeout(1500);
        await shot(p, `desktop-app-layers-after-${label.replace(/[^a-z]/gi,'').toLowerCase()}`, { full: false });
        await el.click({ timeout: 1500 }).catch(()=>{}); // restore
        await p.waitForTimeout(600);
        break;
      }
    } catch (e) { log('layer toggle miss', label, e.message); }
  }

  // /app H1 (different timeframe)
  await p.goto(`${BASE}/fr/app?instrument=XAUUSD&timeframe=H1`, { waitUntil: 'domcontentloaded' });
  await settle(p, 6000);
  await shot(p, 'desktop-app-xauusd-h1-full');

  // Try opening MIA chat
  try {
    for (const t of ['M.I.A', 'MIA', 'Assistant', 'Poser une question', 'Demander']) {
      const b = p.getByRole('button', { name: new RegExp(t, 'i') });
      if (await b.count()) { await b.first().click({ timeout: 1200 }); await p.waitForTimeout(1500); break; }
    }
    await shot(p, 'desktop-app-mia-panel', { full: false });
  } catch (e) { log('mia miss', e.message); }

  // /scanner
  await p.goto(`${BASE}/fr/scanner`, { waitUntil: 'domcontentloaded' });
  await dismiss(p); await settle(p, 5000);
  await shot(p, 'desktop-scanner-full');
  await shot(p, 'desktop-scanner-viewport', { full: false });

  // /zones
  await p.goto(`${BASE}/fr/zones?instrument=XAUUSD&timeframe=M15`, { waitUntil: 'domcontentloaded' });
  await settle(p, 5000);
  await shot(p, 'desktop-zones-full');

  // /actualites (calendar)
  await p.goto(`${BASE}/fr/actualites`, { waitUntil: 'domcontentloaded' });
  await settle(p, 5000);
  await shot(p, 'desktop-actualites-calendar-full');

  await desk.close();

  // ---------- MOBILE 390x844 ----------
  const mob = await browser.newContext({ viewport: { width: 390, height: 844 }, locale: 'fr-FR', isMobile: true, hasTouch: true });
  const m = await mob.newPage();
  const surfaces = [
    ['app', `/fr/app?instrument=XAUUSD&timeframe=M15`],
    ['scanner', `/fr/scanner`],
    ['zones', `/fr/zones?instrument=XAUUSD&timeframe=M15`],
    ['actualites', `/fr/actualites`],
  ];
  for (const [name, url] of surfaces) {
    await m.goto(`${BASE}${url}`, { waitUntil: 'domcontentloaded' });
    await dismiss(m); await settle(m, 5500);
    await shot(m, `mobile-${name}-full`);
  }
  await mob.close();

  await browser.close();
  log('DONE');
}
run().catch((e) => { console.error('FATAL', e); process.exit(1); });
