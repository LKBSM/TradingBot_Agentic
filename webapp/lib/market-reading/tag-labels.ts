/**
 * Vulgarisation des tags descriptifs (Chantier 5.D).
 *
 * Le backend (et le type `MarketReadingConditions.tags`) transporte des codes
 * techniques en snake_case (`bos_confirmed`, `retest_active`…). Ils restent la
 * source de vérité côté données ; SEUL l'affichage est vulgarisé ici, en
 * français simple, niveau 1.5 (description d'un fait de marché, jamais une
 * instruction).
 *
 * Un tag inconnu retombe sur une humanisation générique (underscores → espaces,
 * première lettre capitalisée) plutôt que d'exposer le code brut.
 */

const TAG_LABEL: Record<string, string> = {
  // Tendance / régime
  trend: 'Tendance établie',
  trending: 'Tendance établie',
  ranging: 'Marché en range',
  quiet: 'Marché calme',
  expansion: 'Expansion',
  accumulation: 'Accumulation',
  distribution: 'Distribution',
  // Volatilité
  low_vol: 'Volatilité basse',
  high_vol: 'Volatilité élevée',
  normal_vol: 'Volatilité normale',
  // Structure (BOS / CHOCH)
  bos_confirmed: 'Cassure confirmée',
  bos_pending: 'Cassure en attente',
  choch_confirmed: 'Changement de caractère confirmé',
  choch_pending: 'Changement de caractère en attente',
  // Retest / zones
  retest_active: 'Retest en cours',
  retest_awaiting: 'Retest attendu',
  ob_active: 'Order Block actif',
  fvg_active: 'Fair Value Gap actif',
};

/** Humanise a snake_case fallback: "some_code" → "Some code". */
function humanise(tag: string): string {
  const spaced = tag.replace(/_/g, ' ').trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** Vulgarised, user-facing label for a descriptive tag. */
export function formatTag(tag: string): string {
  return TAG_LABEL[tag] ?? humanise(tag);
}
