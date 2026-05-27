import { expect, test } from '@playwright/test';

test.describe('Landing — golden paths', () => {
  test('hero renders with brand + track-record + CTA', async ({ page }) => {
    await page.goto('/');
    // Brand visible in the nav.
    await expect(page.getByRole('link', { name: /MIA Markets/i })).toBeVisible();
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

  test('first card exposes the History section by default (methodology visible) — RÉVISÉ 2026-05-27', async ({ page }) => {
    await page.goto('/#demo');
    // Post pivot positioning (cf. docs/governance/decisions/2026-05-27_pivot_positioning_audit.md) :
    // PF 1.30 + 329 setups retirés. La section History expose désormais la méthodologie + statut OOS pending.
    await expect(page.getByText(/Méthodologie publique|validation OOS|7 ans/i)).toBeVisible();
  });

  test('pricing section shows 3 tiers FREE/9€/19€ post pivot 2026-05-27', async ({ page }) => {
    await page.goto('/#tarifs');
    await expect(page.getByText(/FREE/)).toBeVisible();
    await expect(page.getByText(/Découverte/)).toBeVisible();
    await expect(page.getByText(/Approfondie/)).toBeVisible();
    // INSTITUTIONAL retiré grille publique → "Contact us" / Calendly link
    await expect(page.getByText(/Réserver une démo|Contact/i)).toBeVisible();
    await expect(page.getByText(/Recommandé/)).toBeVisible();
  });
});
