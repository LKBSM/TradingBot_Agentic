import { getRequestConfig } from 'next-intl/server';

// Curated launch set (décision 2026-07-07): the 8 highest-coverage European
// languages + Arabic. FR stays the default/source locale. Adding a language
// later is a pure additive step: append the code here and drop a
// `messages/<code>.json` alongside the others.
export const SUPPORTED_LOCALES = [
  'fr',
  'en',
  'de',
  'es',
  'it',
  'pt',
  'nl',
  'pl',
  'ar',
] as const;
export type Locale = (typeof SUPPORTED_LOCALES)[number];
export const DEFAULT_LOCALE: Locale = 'fr';

// Right-to-left scripts. Arabic is the only RTL locale in the launch set; the
// layout flips `dir` and swaps the sans stack to an Arabic-capable font for
// these. Keep this list authoritative so both the layout and any future
// direction-aware component read the same source of truth.
export const RTL_LOCALES: readonly Locale[] = ['ar'];

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
  de: 'Deutsch',
  es: 'Español',
  it: 'Italiano',
  pt: 'Português',
  nl: 'Nederlands',
  pl: 'Polski',
  ar: 'العربية',
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
