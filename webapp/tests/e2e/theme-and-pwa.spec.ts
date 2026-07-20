import { expect, test } from '@playwright/test';

test.describe('Theme + PWA — golden paths', () => {
  test('theme menu switches the active data-theme on <html>', async ({ page }) => {
    await page.goto('/');
    const html = page.locator('html');
    // Pick Atelier (light) from the theme menu.
    await page.getByRole('button', { name: /Choisir le thème/i }).click();
    await page.getByRole('menuitemradio', { name: /Atelier/i }).click();
    await expect(html).toHaveAttribute('data-theme', 'atelier');
    // Switch to Terminal (dark) and confirm the attribute flips.
    await page.getByRole('button', { name: /Choisir le thème/i }).click();
    await page.getByRole('menuitemradio', { name: /Terminal/i }).click();
    await expect(html).toHaveAttribute('data-theme', 'terminal');
  });

  test('manifest is served with right content-type', async ({ request }) => {
    const res = await request.get('/manifest.webmanifest');
    expect(res.status()).toBe(200);
    expect(res.headers()['content-type']).toMatch(/manifest/);
    const body = await res.json();
    expect(body.name).toContain('MIA');
    expect(body.display).toBe('standalone');
  });

  test('a non-default active locale is served', async ({ request }) => {
    // i18n is live: `en` (like the other 8 locales) is now served directly,
    // not 302'd to FR. A valid locale root returns 200; an unknown page under
    // it is a normal 404.
    const root = await request.get('/en', { maxRedirects: 0, failOnStatusCode: false });
    expect(root.status()).toBe(200);
    const missing = await request.get('/en/whatever', {
      maxRedirects: 0,
      failOnStatusCode: false,
    });
    expect(missing.status()).toBe(404);
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
