'use client';

import { useRef } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { useLocale, useTranslations } from 'next-intl';
import { Check, Globe } from 'lucide-react';
import {
  DEFAULT_LOCALE,
  LOCALE_LABELS,
  SUPPORTED_LOCALES,
  isSupportedLocale,
  type Locale,
} from '@/i18n';
import { cn } from '@/lib/utils';

/**
 * Language switcher — real dropdown across all 9 active locales
 * (fr/en/de/es/it/pt/nl/pl/ar). Built on a native <details> disclosure so it
 * needs no dropdown dependency, stays keyboard-accessible, and is RTL-safe via
 * logical `start/end` utilities.
 *
 * On selection it:
 *   1. writes the NEXT_LOCALE cookie (next-intl reads this to persist an
 *      explicit choice over the Accept-Language header on later visits),
 *   2. rebuilds the current path under the chosen locale (default locale is
 *      served prefix-less thanks to `localePrefix: 'as-needed'`),
 *   3. navigates there.
 */
export function LocaleToggle() {
  const router = useRouter();
  const pathname = usePathname() ?? '/';
  const active = useLocale() as Locale;
  const t = useTranslations('nav');
  const detailsRef = useRef<HTMLDetailsElement>(null);

  // Strip a leading locale segment (if any) to get the locale-agnostic path.
  const segments = pathname.split('/').filter(Boolean);
  const first = segments[0];
  const basePath =
    first && isSupportedLocale(first) ? '/' + segments.slice(1).join('/') : pathname;

  function selectLocale(locale: Locale) {
    // Close the disclosure immediately for snappy feedback.
    detailsRef.current?.removeAttribute('open');

    // 1 year, root path — matches next-intl's own cookie lifetime.
    document.cookie = `NEXT_LOCALE=${locale}; path=/; max-age=31536000; samesite=lax`;

    const clean = basePath === '' ? '/' : basePath;
    const target = locale === DEFAULT_LOCALE ? clean : `/${locale}${clean === '/' ? '' : clean}`;
    router.push(target);
  }

  return (
    <details ref={detailsRef} className="relative">
      <summary
        className="flex cursor-pointer list-none items-center gap-1.5 rounded-md border border-border/70 px-2 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring [&::-webkit-details-marker]:hidden"
        aria-label={t('language')}
      >
        <Globe className="h-3.5 w-3.5" aria-hidden />
        <span className="uppercase">{active}</span>
      </summary>

      <ul
        role="listbox"
        aria-label={t('language')}
        className="absolute end-0 z-50 mt-1 max-h-[70vh] min-w-[9rem] overflow-auto rounded-md border border-border bg-popover p-1 text-sm text-popover-foreground shadow-md"
      >
        {SUPPORTED_LOCALES.map((locale) => {
          const isActive = locale === active;
          return (
            <li key={locale}>
              <button
                type="button"
                lang={locale}
                dir={locale === 'ar' ? 'rtl' : 'ltr'}
                aria-current={isActive ? 'true' : undefined}
                onClick={() => selectLocale(locale)}
                className={cn(
                  'flex w-full items-center justify-between gap-3 rounded-sm px-2 py-1.5 text-start transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:bg-accent focus-visible:outline-none',
                  isActive && 'font-medium',
                )}
              >
                <span>{LOCALE_LABELS[locale]}</span>
                {isActive ? (
                  <Check className="h-3.5 w-3.5 shrink-0" aria-hidden />
                ) : (
                  <span className="text-[10px] uppercase text-muted-foreground">{locale}</span>
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </details>
  );
}
