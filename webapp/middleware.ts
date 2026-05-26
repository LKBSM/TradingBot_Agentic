import createMiddleware from 'next-intl/middleware';
import { NextRequest, NextResponse } from 'next/server';
import { DEFAULT_LOCALE, SUPPORTED_LOCALES } from './i18n';

// V1 ships FR only. EN/DE/ES translation files stay in the repo as dormant
// infrastructure but their routes must NOT be indexable — empty/stub locale
// pages would trigger SEO duplication penalties. Any incoming /en/* /de/* /es/*
// request is 302-redirected to the FR equivalent (default locale, served
// without prefix thanks to `localePrefix: 'as-needed'`).
const INACTIVE_LOCALES = ['en', 'de', 'es'] as const;

const intlMiddleware = createMiddleware({
  locales: SUPPORTED_LOCALES,
  defaultLocale: DEFAULT_LOCALE,
  localePrefix: 'as-needed',
});

export default function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  for (const inactive of INACTIVE_LOCALES) {
    if (pathname === `/${inactive}` || pathname.startsWith(`/${inactive}/`)) {
      const stripped = pathname.slice(inactive.length + 1) || '/';
      const target = request.nextUrl.clone();
      target.pathname = stripped;
      return NextResponse.redirect(target, 302);
    }
  }

  return intlMiddleware(request);
}

export const config = {
  matcher: ['/((?!api|_next|_vercel|.*\\..*).*)'],
};
