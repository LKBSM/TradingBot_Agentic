import { test, expect, type Page } from '@playwright/test';
import path from 'node:path';
import { dismissCookieBanner } from './utils';
import { mockAllApis } from './ds-mock';
import { SAMPLE_CALENDAR_EVENT_ID } from '../../lib/ds-samples';

/**
 * DS-1 coverage — a faithful snapshot of the CURRENT UI for Claude Design:
 * every route × the 4 themes × 2 viewports, rendered with the frozen real-data
 * samples (no backend, no source changes). This is the "recopy my current UI"
 * reference set; from it the founder edits freely.
 *
 * Runs against `next dev` (the gallery is dev-only; product pages self-fetch and
 * are answered by the mock). Themes are forced via next-themes' localStorage key.
 */

const OUT = path.resolve(__dirname, '../../../docs/audits/ds-1/coverage');
const THEMES = ['terminal', 'atelier', 'schema', 'ardoise'] as const;
const VIEWPORTS = [
  { tag: 'desktop', width: 1280, height: 800 },
  { tag: 'mobile', width: 390, height: 844 },
];

type Route = { name: string; url: string; hasChart?: boolean; ready?: string };
const ROUTES: Route[] = [
  // marketing / auth / legal (site group)
  { name: 'accueil', url: '/' },
  { name: 'abonnement', url: '/abonnement' },
  { name: 'methodology', url: '/methodology' },
  { name: 'conditions', url: '/conditions' },
  { name: 'confidentialite', url: '/confidentialite' },
  { name: 'connexion', url: '/connexion' },
  { name: 'inscription', url: '/inscription' },
  { name: 'inscription-google', url: '/inscription/google' },
  { name: 'mot-de-passe-oublie', url: '/mot-de-passe-oublie' },
  { name: 'mot-de-passe-oublie-confirmer', url: '/mot-de-passe-oublie/confirmer' },
  { name: 'verifier-email', url: '/verifier-email' },
  // product (behind SubscriptionGate — mock grants access)
  { name: 'app', url: '/app?instrument=XAUUSD&timeframe=H4', hasChart: true },
  { name: 'zones', url: '/zones?instrument=XAUUSD&timeframe=H4' },
  { name: 'scanner', url: '/scanner' },
  { name: 'scanner-decrire', url: '/scanner/decrire' },
  { name: 'actualites', url: '/actualites', ready: '.calm-grid' },
  { name: 'actualites-fiche', url: `/actualites/${SAMPLE_CALENDAR_EVENT_ID}`, ready: '.cal-page' },
  { name: 'compte', url: '/compte' },
];

async function settle(page: Page, r: Route) {
  await dismissCookieBanner(page);
  await page.locator('.animate-spin').first().waitFor({ state: 'detached', timeout: 12_000 }).catch(() => {});
  if (r.ready) {
    await page.locator(r.ready).first().waitFor({ state: 'visible', timeout: 15_000 }).catch(() => {});
  }
  if (r.hasChart) {
    await page.locator('canvas').first().waitFor({ state: 'visible', timeout: 15_000 }).catch(() => {});
    await page.waitForTimeout(1_500);
  }
  await page.waitForTimeout(600);
}

for (const theme of THEMES) {
  test.describe(`theme:${theme}`, () => {
    for (const vp of VIEWPORTS) {
      for (const r of ROUTES) {
        test(`${r.name} · ${vp.tag}`, async ({ page }) => {
          test.setTimeout(120_000);
          // Force the theme before any app script runs (next-themes reads it pre-paint).
          await page.addInitScript((t) => {
            try { localStorage.setItem('theme', t as string); } catch {}
          }, theme);
          await mockAllApis(page);
          await page.setViewportSize({ width: vp.width, height: vp.height });
          await page.goto(r.url, { waitUntil: 'domcontentloaded' });
          await settle(page, r);
          await expect(page.locator('body')).toBeVisible();
          await page.screenshot({ path: path.join(OUT, `${r.name}--${theme}--${vp.tag}.png`), fullPage: true });
        });
      }
    }
  });
}
