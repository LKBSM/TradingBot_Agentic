import { expect, test } from '@playwright/test';

/**
 * UI-2b — App centre conformity. These assert the structural facts that hold
 * WITHOUT the data backend (which the e2e server does not provide): from 768px
 * up the centre is the reference single column (no tabs), the rail is present,
 * no horizontal scroll, no raw i18n key leaks; the chat is docked ≥1280 and an
 * off-canvas drawer between 768 and 1279; below 768 the tabbed MobileWorkspace
 * takes over. The 5 data panels + legalbar are covered by the component unit
 * tests + the mock-data visual check in the audit report.
 */

// Signature of an unresolved i18n key rendered to screen (e.g. legal.earlyAccessBadge).
const RAW_KEY_SCAN = `(() => {
  const bad = [];
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = w.nextNode())) {
    const t = (n.textContent || '').trim();
    if (!t || t.includes('/') || t.includes('@') || t.includes(':')) continue;
    if (/^[a-z][a-zA-Z0-9]*(\\.[a-zA-Z][a-zA-Z0-9]*)+$/.test(t)) bad.push(t);
  }
  return bad;
})()`;

for (const width of [1280, 1024]) {
  test(`≥768 (${width}px): single-column centre, no tabs, no raw key, no h-scroll`, async ({ page }) => {
    await page.setViewportSize({ width, height: 800 });
    await page.goto('/app');

    await expect(
      page.getByRole('complementary', { name: /combinaisons disponibles/i }),
    ).toBeVisible();

    // No tab bar in the centre — MobileWorkspace (tabs) is phone-only (<768).
    await expect(page.getByRole('tab')).toHaveCount(0);

    // No unresolved i18n key on screen.
    const raw = await page.evaluate(RAW_KEY_SCAN);
    expect(raw).toEqual([]);

    // No horizontal overflow.
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(1);
  });
}

test('1024px: the chat is an off-canvas drawer opened by the FAB', async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 800 });
  await page.goto('/app');
  const fab = page.locator('.chat-fab');
  await expect(fab).toBeVisible();
  await fab.click();
  await expect(page.locator('.chatcol.open')).toBeVisible();
});

test('≥1280px: the chat is docked (no FAB)', async ({ page }) => {
  await page.setViewportSize({ width: 1360, height: 800 });
  await page.goto('/app');
  await expect(page.locator('.chat-fab')).toBeHidden();
  await expect(page.locator('.chatcol')).toBeVisible();
});

test('<768px: the tabbed MobileWorkspace takes over', async ({ page }) => {
  await page.setViewportSize({ width: 720, height: 900 });
  await page.goto('/app');
  // Locale-independent: the phone workspace shows its 3-tab bar (Marchés ·
  // Lecture · Chat). The shell rail + docked chat are hidden here.
  await expect(page.getByRole('tab')).toHaveCount(3);
  await expect(page.locator('.chat-fab')).toBeHidden();
});
