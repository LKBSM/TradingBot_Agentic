/**
 * Descriptive-tag display (Chantier 5.D → i18n Étape 3).
 *
 * The backend (and `MarketReadingConditions.tags`) carries technical snake_case
 * codes (`bos_confirmed`, `retest_active`…). They stay the data source of truth;
 * the vulgarised, user-facing labels now live in the `reading.tags.*` message
 * namespace and are resolved via `useTranslations` at the component boundary
 * (see `ConditionsSection`). A code with no mapped label falls back to the
 * generic humanisation below (underscores → spaces, capitalised) rather than
 * exposing the raw code.
 */

/**
 * Humanise a snake_case fallback for an UNMAPPED tag.
 *
 * Last-resort path only: every tag the backend actually emits is mapped in the
 * `reading.tags` messages. For a genuinely unknown code we capitalise and
 * de-snake it ("some_code" → "Some code") — language-neutral, so no client-facing
 * term is hard-coded here; the real fix for a recurring tag is to map it.
 */
export function humaniseTag(tag: string): string {
  const spaced = tag.replace(/_/g, ' ').trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}
