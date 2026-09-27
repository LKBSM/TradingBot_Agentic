import { getRequestConfig } from 'next-intl/server';

// Périmètre de lancement (décision fondateur, 2026-09-27) : FRANÇAIS et
// ANGLAIS uniquement. Les sept autres langues (de, es, it, pt, nl, pl, ar)
// étaient traduites, mais elles multipliaient par neuf la surface à relire et
// à garder honnête à chaque changement de texte — pour un marché cible qui est
// le Canada et les États-Unis. Elles reviendront plus tard.
//
// FR reste la locale par défaut et la langue source. Rajouter une langue est un
// geste purement additif : ajouter son code ici et déposer le
// `messages/<code>.json` correspondant.
export const SUPPORTED_LOCALES = ['fr', 'en'] as const;
export type Locale = (typeof SUPPORTED_LOCALES)[number];
export const DEFAULT_LOCALE: Locale = 'fr';

// Écritures de droite à gauche. L'arabe était la seule du lot et il est retiré,
// donc la liste est vide — mais la mécanique reste en place : le jour où une
// langue RTL revient, il suffit de l'ajouter ici et la mise en page suivra.
export const RTL_LOCALES: readonly Locale[] = [];

export function isRtl(locale: string): boolean {
  return RTL_LOCALES.includes(locale as Locale);
}

export function isSupportedLocale(value: string): value is Locale {
  return SUPPORTED_LOCALES.includes(value as Locale);
}

/** Human-readable, self-referential label for each locale (used by the
 * language switcher — always shown in the language's own script). */
export const LOCALE_LABELS: Record<Locale, string> = {
  fr: 'Français',
  en: 'English',
};

/**
 * next-intl 3.22+ pattern: read `requestLocale` (async) instead of the
 * deprecated `locale` arg, fall back to the default if unset/unsupported,
 * and return both `locale` and `messages` so the rest of the runtime can
 * read them synchronously.
 */
export default getRequestConfig(async ({ requestLocale }) => {
  const requested = await requestLocale;
  const locale: Locale =
    requested && isSupportedLocale(requested) ? requested : DEFAULT_LOCALE;
  return {
    locale,
    messages: (await import(`./messages/${locale}.json`)).default,
  };
});
