'use client';

import { useCallback } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import {
  DEFAULT_LOCALE,
  isSupportedLocale,
  type Locale,
} from '@/i18n';

/**
 * THE single locale-switch action, shared by every language selector (the site
 * `LocaleToggle` in the nav AND the « Langue » section in Réglages) so there is
 * one source of truth for the locale — never two competing mechanisms.
 *
 * On selection it:
 *   1. writes the `NEXT_LOCALE` cookie (1 year) — next-intl reads it to persist
 *      an EXPLICIT choice over the `Accept-Language` header on later visits, so
 *      the chosen language is restored on return and wins over the browser;
 *   2. rebuilds the CURRENT path under the chosen locale (default locale is
 *      served prefix-less via `localePrefix: 'as-needed'`), keeping the user on
 *      the same page with the same context;
 *   3. navigates there with a client `router.push` — no full page reload.
 */
export function useLocaleSwitch(): (locale: Locale) => void {
  const router = useRouter();
  const pathname = usePathname() ?? '/';

  return useCallback(
    (locale: Locale) => {
      // 1 year, root path — matches next-intl's own cookie lifetime.
      document.cookie = `NEXT_LOCALE=${locale}; path=/; max-age=31536000; samesite=lax`;

      // Strip a leading locale segment (if any) to get the locale-agnostic path.
      const segments = pathname.split('/').filter(Boolean);
      const first = segments[0];
      const basePath =
        first && isSupportedLocale(first) ? '/' + segments.slice(1).join('/') : pathname;

      const clean = basePath === '' ? '/' : basePath;
      const target =
        locale === DEFAULT_LOCALE ? clean : `/${locale}${clean === '/' ? '' : clean}`;
      router.push(target);
    },
    [router, pathname],
  );
}
