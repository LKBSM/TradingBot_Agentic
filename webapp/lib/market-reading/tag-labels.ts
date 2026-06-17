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
  // Tendance / régime (libellés "legacy" présents dans les mocks/fixtures)
  trend: 'Tendance établie',
  trending: 'Tendance établie',
  ranging: 'Marché en range',
  quiet: 'Marché calme',
  expansion: 'Expansion',
  accumulation: 'Accumulation',
  distribution: 'Distribution',
  // Volatilité (libellés "legacy")
  low_vol: 'Volatilité basse',
  high_vol: 'Volatilité élevée',
  normal_vol: 'Volatilité normale',
  // Structure (BOS / CHOCH) — libellés "legacy"
  bos_confirmed: 'Cassure confirmée',
  bos_pending: 'Cassure en attente',
  choch_confirmed: 'Changement de caractère confirmé',
  choch_pending: 'Changement de caractère en attente',
  // Retest / zones
  retest_active: 'Retest en cours',
  retest_awaiting: 'Retest attendu',
  ob_active: 'Order Block actif',
  fvg_active: 'Fair Value Gap actif',

  // ── Tags réellement émis par le backend (_build_tags, market_reading_mappers) ──
  // Sans ces entrées, ils retombaient sur humanise() → libellés anglophones
  // ("Trend bearish", "Volatility elevated", "Phase expansion"…). Fuite i18n.
  // Tendance : `trend_<value>`
  trend_bullish: 'Tendance haussière',
  trend_bearish: 'Tendance baissière',
  trend_neutral: 'Tendance neutre',
  trend_ranging: 'Marché en range',
  // Volatilité : `volatility_<value>`
  volatility_low: 'Volatilité basse',
  volatility_normal: 'Volatilité normale',
  volatility_elevated: 'Volatilité élevée',
  // Phase de marché : `phase_<value>`
  phase_accumulation: 'Phase d’accumulation',
  phase_distribution: 'Phase de distribution',
  phase_trend: 'Phase de tendance',
  phase_ranging: 'Phase de range',
  phase_expansion: 'Phase d’expansion',
  // Structure récente : `bos_recent_<dir>` / `choch_recent_<dir>`
  bos_recent_bullish: 'Cassure haussière récente',
  bos_recent_bearish: 'Cassure baissière récente',
  choch_recent_bullish: 'Changement de caractère haussier',
  choch_recent_bearish: 'Changement de caractère baissier',
  // Retest en cours
  retest_in_progress: 'Retest en cours',
  // Confluence multi-timeframe
  mtf_aligned: 'Timeframes alignés',
  mtf_divergent: 'Timeframes divergents',
  mtf_mixed: 'Timeframes mixtes',
};

/**
 * Humanise a snake_case fallback for an UNKNOWN tag, in French.
 *
 * Last-resort path only: every tag the backend actually emits is mapped in
 * TAG_LABEL above. For a genuinely unknown code we capitalise and de-snake it
 * ("some_code" → "Some code") — kept generic so no client-facing English term
 * is hard-coded here, but the real fix for a recurring tag is to map it.
 */
function humanise(tag: string): string {
  const spaced = tag.replace(/_/g, ' ').trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** Vulgarised, user-facing label for a descriptive tag. */
export function formatTag(tag: string): string {
  return TAG_LABEL[tag] ?? humanise(tag);
}
