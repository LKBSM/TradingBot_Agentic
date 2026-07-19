import type { Metadata, Viewport } from 'next';
import { Inter, Noto_Sans_Arabic } from 'next/font/google';
import { NextIntlClientProvider } from 'next-intl';
import { getMessages, getTranslations } from 'next-intl/server';
import { notFound } from 'next/navigation';
import '../globals.css';
import { Nav } from '@/components/Nav';
import { Footer } from '@/components/Footer';
import { SkipLink } from '@/components/a11y/SkipLink';
import { ChatPanel } from '@/components/chat/ChatPanel';
import { ChatProvider } from '@/components/chat/ChatProvider';
import { ChartViewProvider } from '@/lib/chart/viewState';
import { CookieBanner } from '@/components/compliance/CookieBanner';
import { JsonLd, softwareApplicationLd } from '@/components/seo/JsonLd';
import { ThemeProvider } from '@/components/theme-provider';
import { TooltipProvider } from '@/components/ui/tooltip';
import { AuthProvider } from '@/lib/auth/store';
import { BRAND_NAME } from '@/lib/brand';
import {
  DEFAULT_LOCALE,
  SUPPORTED_LOCALES,
  isRtl,
  isSupportedLocale,
  type Locale,
} from '../../i18n';

// Inter, variable axis weight 100..900, only the Latin subset (no CJK/Cyrillic
// shipped). Latin-ext kept for accented FR characters. `display: 'swap'`
// avoids invisible text during the font swap, `preload: true` puts the
// woff2 file in the HTML <link rel="preload">.
const inter = Inter({
  subsets: ['latin', 'latin-ext'],
  display: 'swap',
  preload: true,
  variable: '--font-sans',
});

// Arabic-capable font, exposed as --font-arabic. The Latin `Inter` subset has
// no Arabic glyphs, so RTL locales fall back to it via a globals.css rule that
// swaps the body font stack when `html[dir="rtl"]`. Not preloaded — only ar
// visitors pay for it, and only after the (already-swapped) Inter.
const notoArabic = Noto_Sans_Arabic({
  subsets: ['arabic'],
  display: 'swap',
  preload: false,
  variable: '--font-arabic',
});

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://mia.markets';

// BCP-47 / Open Graph locale codes (language_TERRITORY) for each launch locale.
// hreflang uses the plain language subtag from SUPPORTED_LOCALES; Open Graph
// wants the territory-qualified form, so we keep an explicit map (no reliable
// language→country inference exists — pt could be pt_PT or pt_BR).
const OG_LOCALES: Record<Locale, string> = {
  fr: 'fr_FR',
  en: 'en_US',
  de: 'de_DE',
  es: 'es_ES',
  it: 'it_IT',
  pt: 'pt_PT',
  nl: 'nl_NL',
  pl: 'pl_PL',
  ar: 'ar_AR',
};

// `localePrefix: 'as-needed'` (see middleware): the default locale (fr) is
// served prefix-less at `/`, every other locale under `/<code>`.
function localePath(locale: Locale): string {
  return locale === DEFAULT_LOCALE ? '/' : `/${locale}`;
}

// hreflang alternates for every launch locale + an x-default pointing at the
// default (prefix-less) locale. Keyed by the plain language subtag, which is
// what next-intl serves and what Google matches against.
function hreflangLanguages(): Record<string, string> {
  const languages: Record<string, string> = {};
  for (const loc of SUPPORTED_LOCALES) {
    languages[loc] = localePath(loc);
  }
  languages['x-default'] = localePath(DEFAULT_LOCALE);
  return languages;
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale: raw } = await params;
  const locale: Locale = isSupportedLocale(raw) ? raw : DEFAULT_LOCALE;
  const t = await getTranslations({ locale, namespace: 'seo' });

  return {
    metadataBase: new URL(SITE_URL),
    title: {
      default: t('title'),
      template: `%s · ${BRAND_NAME}`,
    },
    description: t('description'),
    applicationName: BRAND_NAME,
    authors: [{ name: BRAND_NAME }],
    robots: { index: true, follow: true },
    alternates: {
      // Canonical is the current locale's own path — each localized page is
      // its own canonical, cross-linked to the others via hreflang below.
      canonical: localePath(locale),
      languages: hreflangLanguages(),
    },
    openGraph: {
      type: 'website',
      locale: OG_LOCALES[locale],
      // Every other launch locale, so crawlers know the equivalent variants.
      alternateLocale: SUPPORTED_LOCALES.filter((l) => l !== locale).map(
        (l) => OG_LOCALES[l],
      ),
      url: `${SITE_URL}${localePath(locale) === '/' ? '' : localePath(locale)}`,
      siteName: BRAND_NAME,
      title: t('ogTitle'),
      description: t('ogDescription'),
    },
    twitter: {
      card: 'summary_large_image',
      title: t('twitterTitle'),
      description: t('twitterDescription'),
    },
    // PWA install: iOS Safari reads these to badge the home-screen icon and
    // hide the browser chrome when launched standalone.
    appleWebApp: {
      capable: true,
      title: 'MIA',
      statusBarStyle: 'black-translucent',
    },
    formatDetection: {
      telephone: false,
    },
  };
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  viewportFit: 'cover',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0a0f1c' },
  ],
};

export default async function LocaleLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  if (!SUPPORTED_LOCALES.includes(locale as (typeof SUPPORTED_LOCALES)[number])) {
    notFound();
  }
  const messages = await getMessages();
  const dir = isRtl(locale) ? 'rtl' : 'ltr';
  return (
    <html
      lang={locale}
      dir={dir}
      className={`${inter.variable} ${notoArabic.variable}`}
      suppressHydrationWarning
    >
      <body className="flex min-h-screen flex-col bg-background font-sans antialiased">
        <SkipLink />
        <JsonLd data={softwareApplicationLd} />
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <NextIntlClientProvider messages={messages}>
            <TooltipProvider delayDuration={200}>
              <AuthProvider>
                <ChatProvider>
                  {/* Chart view state (layer/zone visibility, focus) is hoisted
                      here — above /app AND /zones — so an action taken on one
                      surface (e.g. masking a zone from /zones) is reflected on the
                      chart. Display-only; it never holds or touches detection. */}
                  <ChartViewProvider>
                    <Nav />
                    {/* flex-1 fills the space between the sticky header and the
                        footer without a hard-coded height guess. */}
                    <main id="main" className="flex-1">
                      {children}
                    </main>
                    <Footer />
                    <ChatPanel />
                    <CookieBanner />
                  </ChartViewProvider>
                </ChatProvider>
              </AuthProvider>
            </TooltipProvider>
          </NextIntlClientProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
