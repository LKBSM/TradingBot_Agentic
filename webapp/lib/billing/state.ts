/**
 * L'état d'abonnement tel que le produit le lit — une seule dérivation.
 *
 * Elle vivait à l'intérieur de `SubscriptionPanel`. La page `/compte` doit dire
 * la même chose que la page `/abonnement` : deux dérivations séparées finiraient
 * par diverger, et l'une des deux mentirait au client sur ce qu'il paie.
 *
 * Ce module ne fait AUCUN appel réseau et ne rend rien : il traduit seulement la
 * réponse de Stripe (`status` + `cancel_at_period_end`) en un état que l'interface
 * sait nommer.
 */
import type { Subscription } from './api-client';

/** Les statuts Stripe qui valent accès immédiat. */
export const ACTIVE_STATUSES = new Set(['active', 'trialing']);

/** Les états côté produit (PAY-1). */
export type SubState = 'none' | 'active' | 'canceling' | 'grace' | 'suspended' | 'expired';

export function deriveState(sub: Subscription | null): SubState {
  const status = sub?.status ?? null;
  if (!status) return 'none';
  if (ACTIVE_STATUSES.has(status)) {
    return sub?.cancel_at_period_end ? 'canceling' : 'active';
  }
  if (status === 'past_due') return 'grace';
  if (status === 'suspended') return 'suspended';
  return 'expired';
}

/**
 * L'état donne-t-il accès au produit en ce moment ?
 *
 * `grace` (paiement échoué) compte : on ne coupe pas l'accès au premier échec de
 * prélèvement, le client a le temps de mettre sa carte à jour.
 */
export function hasAccessState(s: SubState): boolean {
  return s === 'active' || s === 'canceling' || s === 'grace';
}

/** Clé i18n sous `billing.status.*` pour un état donné. */
export function statusKey(state: SubState): string {
  switch (state) {
    case 'active':
    case 'canceling':
      return 'status.active';
    case 'grace':
      return 'status.pastDue';
    case 'suspended':
      return 'status.suspended';
    case 'expired':
      return 'status.expired';
    default:
      return 'status.none';
  }
}
