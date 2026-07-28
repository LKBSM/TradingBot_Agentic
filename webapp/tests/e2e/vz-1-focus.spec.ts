import { expect, test, type Page } from '@playwright/test';
import { dismissCookieBanner } from './utils';

/**
 * VZ-1 — the unified click→select gesture. It is observable at the DOM level via
 * `aria-pressed` on the panel rows, because the selection is a SINGLE piece of
 * provider state shared product-wide. The landing multi-market gallery renders
 * StructureSection under that shared ChartViewProvider with fixture data, so
 * this runs WITHOUT a backend. The camera framing math itself is covered by the
 * focusController unit tests (no canvas assertions here).
 */

async function openFirstStructure(page: Page) {
  await page.goto('/#multi-marche', { waitUntil: 'domcontentloaded' });
  await dismissCookieBanner(page);
  const trigger = page.getByRole('button', { name: /Structure de marché/i }).first();
  // Tolerate a cold `next dev` first-compile (can exceed the default action
  // timeout on a loaded machine) before interacting — the gesture itself is fast.
  await trigger.waitFor({ state: 'visible', timeout: 60_000 });
  await trigger.scrollIntoViewIfNeeded();
  await trigger.click();
  return trigger;
}

/** First selectable zone row (OB « · actif » or FVG « active »). */
function firstZone(page: Page) {
  return page.getByRole('button', { name: /· actif|active/i }).first();
}

test.describe('VZ-1 unified selection @ 1280×800', () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test('clicking a zone selects it; re-clicking toggles it off', async ({ page }) => {
    await openFirstStructure(page);
    const zone = firstZone(page);
    await expect(zone).toHaveAttribute('aria-pressed', 'false');
    await zone.click();
    await expect(zone).toHaveAttribute('aria-pressed', 'true');
    await zone.click();
    await expect(zone).toHaveAttribute('aria-pressed', 'false');
  });

  test('Escape deselects the active element', async ({ page }) => {
    await openFirstStructure(page);
    const zone = firstZone(page);
    await zone.click();
    await expect(zone).toHaveAttribute('aria-pressed', 'true');
    await page.keyboard.press('Escape');
    await expect(zone).toHaveAttribute('aria-pressed', 'false');
  });

  test('single selection across families: a BOS event deselects the active zone', async ({
    page,
  }) => {
    await openFirstStructure(page);
    const zone = firstZone(page);
    // Exact name so we hit the selectable EVENT row (aria-label = bosLabel), not
    // the glossary ⓘ button whose label is « … (BOS) — définition ».
    const bos = page
      .getByRole('button', { name: 'Cassure de structure (BOS)', exact: true })
      .first();
    await zone.click();
    await expect(zone).toHaveAttribute('aria-pressed', 'true');
    await bos.click();
    await expect(bos).toHaveAttribute('aria-pressed', 'true');
    // Only one element is selected product-wide — the zone let go.
    await expect(zone).toHaveAttribute('aria-pressed', 'false');
  });
});

test.describe('VZ-1 gesture on a narrow window @ 390×844', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('the same click→select gesture works under 768px', async ({ page }) => {
    await openFirstStructure(page);
    const zone = firstZone(page);
    await zone.click();
    await expect(zone).toHaveAttribute('aria-pressed', 'true');
    await page.keyboard.press('Escape');
    await expect(zone).toHaveAttribute('aria-pressed', 'false');
  });
});
