import * as React from 'react';
import { useLocale } from 'next-intl';
import { DEFAULT_LOCALE } from '@/i18n';

/**
 * Prefix an internal absolute path with the active locale (NAV-06).
 *
 * `localePrefix: 'as-needed'` (see middleware/i18n): the default locale (fr) is
 * served WITHOUT a prefix, every other locale under `/<code>`. Hash-only and
 * external hrefs pass through untouched — a same-page anchor must not be
 * locale-prefixed. Idempotent-safe callers only pass unprefixed app paths.
 */
export function localizeHref(href: string, locale: string): string {
  if (!href.startsWith('/')) return href;
  return locale === DEFAULT_LOCALE ? href : `/${locale}${href}`;
}

/**
 * Hook form: returns a stable `localize(href)` bound to the active locale. Use
 * for `<Link href={lh('/app')}>` and `router.push(lh('/compte'))` so navigation
 * keeps the reader's locale instead of dropping them onto the default one.
 */
export function useLocalizedHref(): (href: string) => string {
  const locale = useLocale();
  return React.useCallback((href: string) => localizeHref(href, locale), [locale]);
}
