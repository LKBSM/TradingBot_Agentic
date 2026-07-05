import type { Page } from '@playwright/test';

/**
 * Ferme la bannière cookies comme le ferait un utilisateur (clic « Tout
 * refuser » — les tests n'ont besoin d'aucun consentement).
 *
 * Indispensable sur le viewport mobile : la bannière est `fixed inset-x-2
 * bottom-2 z-50` (pleine largeur en bas) et intercepte les clics sur tout
 * élément proche du bord inférieur (accordéons, input du chat…). Sur desktop
 * elle est bottom-right max-w-md et gêne rarement, mais on la ferme partout
 * pour un comportement uniforme.
 */
export async function dismissCookieBanner(page: Page): Promise<void> {
  const reject = page.getByRole('button', { name: /Tout refuser/i });
  try {
    await reject.click({ timeout: 3_000 });
  } catch {
    // Déjà fermée (consentement mémorisé) ou pas encore montée — sans impact.
  }
}
