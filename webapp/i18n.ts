import { getRequestConfig } from 'next-intl/server';

// Commercial launch set (décision 2026-09-03, mission I18N-1): the product ships
// in THREE languages and three only — French (default/source), English, Spanish
// (es-ES). The former 9-locale set (de/it/pt/nl/pl/ar) was retired: a locale
// served with partially-translated content is an interface lie. Adding a
// language later is a pure additive step: append the code here and drop a
// `messages/<code>.json` alongside the others — the switcher, routing and the
// parity guard all derive from THIS list, so there is one source of truth.
export const SUPPORTED_LOCALES = ['fr', 'en', 'es'] as const;
export type Locale = (typeof SUPPORTED_LOCALES)[number];
export const DEFAULT_LOCALE: Locale = 'fr';

// Right-to-left scripts. The launch set (fr/en/es) is entirely LTR, so this list
// is empty. It stays as the authoritative seam: reintroducing an RTL locale
// (e.g. Arabic) means adding its code here AND its `messages/<code>.json`, and
// both the layout `dir` and any direction-aware component read from here.
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
  es: 'Español',
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
    // A missing key must NEVER fall back silently to another language — that is
    // an interface lie. next-intl does not cross-fall-back between locales by
    // default; here we make the anomaly LOUD instead of invisible:
    //   · the CI parity guard (locale-parity.test.ts) already fails the build on
    //     any missing/orphan key — that is the hard gate;
    //   · at runtime, `onError` logs the anomaly so «l'impossible» is journalised
    //     in production instead of passing unseen;
    //   · `getMessageFallback` renders a bracketed marker in dev so a missing key
    //     is spotted IMMEDIATELY on screen, and the raw key path in prod — never
    //     a wrong-language string invented behind the user's back.
    onError(error) {
      // eslint-disable-next-line no-console
      console.error(`[i18n:${locale}] ${error.message}`);
    },
    getMessageFallback({ namespace, key }) {
      const path = [namespace, key].filter(Boolean).join('.');
      return process.env.NODE_ENV === 'development' ? `⟦${path}⟧` : path;
    },
  };
});
