import { expect, test } from '@playwright/test';

test.describe('Landing — golden paths', () => {
  test('hero renders with brand + track-record + CTA', async ({ page }) => {
    await page.goto('/');
    // Brand visible in the nav.
    await expect(page.getByRole('link', { name: /M\.I\.A\. Markets/i })).toBeVisible();
    // Hero headline.
    await expect(
      page.getByRole('heading', { level: 1, name: /Comprenez le marché/i }),
    ).toBeVisible();
    // Track record pépite.
    await expect(page.getByText(/Track record honnête/i)).toBeVisible();
    await expect(page.getByText(/IC 95/)).toBeVisible();
    // Primary CTA.
    await expect(page.getByRole('link', { name: /Voir une lecture/i })).toBeVisible();
  });

  test('three demo cards are rendered with hero + verdicts', async ({ page }) => {
    await page.goto('/#demo');
    // 3 verdicts (one per signal). Each verdict is an h2 in the card hero.
    const verdicts = page.getByRole('heading', { level: 2 });
    // Expect at least 4 (demo section title + 3 cards).
    await expect(verdicts.first()).toBeVisible();
    await expect(page.getByText(/Lecture haussière sur l'or/i)).toBeVisible();
    await expect(page.getByText(/Lecture baissière sur l'euro/i)).toBeVisible();
    await expect(page.getByText(/Lecture neutre sur l'or/i)).toBeVisible();
  });

  test('first card exposes the History section by default (PF visible)', async ({ page }) => {
    await page.goto('/#demo');
    // 329 setups + PF 1.30 should be visible without any interaction.
    await expect(page.getByText(/329 setups/i)).toBeVisible();
    await expect(page.getByText(/1,30/)).toBeVisible();
  });

  test('pricing section shows 4 tiers + decoy + recommended badges', async ({ page }) => {
    await page.goto('/#tarifs');
    await expect(page.getByRole('heading', { name: /Trois formules retail/i })).toBeVisible();
    await expect(page.getByText('Découverte')).toBeVisible();
    await expect(page.getByText('Analyste')).toBeVisible();
    await expect(page.getByText('Stratège')).toBeVisible();
    await expect(page.getByText('Institutionnel')).toBeVisible();
    await expect(page.getByText(/Recommandé/)).toBeVisible();
  });
});
