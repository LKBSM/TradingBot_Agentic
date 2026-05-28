import { expect, test } from '@playwright/test';

test.describe('Theme + PWA — golden paths', () => {
  test('theme toggle switches html.dark class', async ({ page }) => {
    await page.goto('/');
    const html = page.locator('html');
    const initialDark = await html.evaluate((el) => el.classList.contains('dark'));
    await page
      .getByRole('button', { name: /(Activer le mode (clair|sombre))/i })
      .click();
    // Wait one frame for next-themes to settle.
    await page.waitForTimeout(150);
    const flipped = await html.evaluate((el) => el.classList.contains('dark'));
    expect(flipped).not.toBe(initialDark);
  });

  test('manifest is served with right content-type', async ({ request }) => {
    const res = await request.get('/manifest.webmanifest');
    expect(res.status()).toBe(200);
    expect(res.headers()['content-type']).toMatch(/manifest/);
    const body = await res.json();
    expect(body.name).toContain('MIA');
    expect(body.display).toBe('standalone');
  });

  test('inactive locale routes 302 to FR equivalent', async ({ request }) => {
    const res = await request.get('/en/whatever', {
      maxRedirects: 0,
      failOnStatusCode: false,
    });
    expect(res.status()).toBe(302);
  });

  test('icon endpoints respond 200 with image content-type', async ({ request }) => {
    const icon = await request.get('/icon');
    expect(icon.status()).toBe(200);
    expect(icon.headers()['content-type']).toMatch(/image\/png/);

    const apple = await request.get('/apple-icon');
    expect(apple.status()).toBe(200);
    expect(apple.headers()['content-type']).toMatch(/image\/png/);
  });
});
