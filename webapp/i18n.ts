import { getRequestConfig } from 'next-intl/server';

export const SUPPORTED_LOCALES = ['fr', 'en', 'de', 'es'] as const;
export type Locale = (typeof SUPPORTED_LOCALES)[number];
export const DEFAULT_LOCALE: Locale = 'fr';

/**
 * next-intl 3.22+ pattern: read `requestLocale` (async) instead of the
 * deprecated `locale` arg, fall back to the default if unset/unsupported,
 * and return both `locale` and `messages` so the rest of the runtime can
 * read them synchronously.
 */
export default getRequestConfig(async ({ requestLocale }) => {
  const requested = await requestLocale;
  const locale: Locale =
    requested && SUPPORTED_LOCALES.includes(requested as Locale)
      ? (requested as Locale)
      : DEFAULT_LOCALE;
  return {
    locale,
    messages: (await import(`./messages/${locale}.json`)).default,
  };
});
