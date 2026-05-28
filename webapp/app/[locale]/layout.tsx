import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import { NextIntlClientProvider } from 'next-intl';
import { getMessages } from 'next-intl/server';
import { notFound } from 'next/navigation';
import '../globals.css';
import { Nav } from '@/components/Nav';
import { Footer } from '@/components/Footer';
import { SkipLink } from '@/components/a11y/SkipLink';
import { PlausibleScript } from '@/components/analytics/PlausibleScript';
import { ChatPanel } from '@/components/chat/ChatPanel';
import { ChatProvider } from '@/components/chat/ChatProvider';
import { CookieBanner } from '@/components/compliance/CookieBanner';
import { JsonLd, softwareApplicationLd } from '@/components/seo/JsonLd';
import { ThemeProvider } from '@/components/theme-provider';
import { TooltipProvider } from '@/components/ui/tooltip';
import { SUPPORTED_LOCALES } from '../../i18n';

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

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://mia.markets';

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: 'MIA Markets — Indicateur de marché conversationnel',
    template: '%s · MIA Markets',
  },
  description:
    "MIA Markets (Multi-asset Intelligence Assistant) — Lecture de marché XAU/USD et FX, expliquée par un quant. Analyses contextuelles sourcées, posture éducative, conforme UE 2024/2811.",
  applicationName: 'MIA Markets',
  authors: [{ name: 'MIA Markets' }],
  robots: { index: true, follow: true },
  alternates: {
    canonical: '/',
    // EN is filed but inactive (302 → FR). hreflang signals the future
    // bilingual ambition to crawlers without serving duplicate content.
    languages: {
      'fr-FR': '/',
      'x-default': '/',
    },
  },
  openGraph: {
    type: 'website',
    locale: 'fr_FR',
    url: SITE_URL,
    siteName: 'MIA Markets',
    title: 'MIA Markets — Indicateur de marché conversationnel',
    description:
      'Lectures algorithmiques contextuelles pour XAU/USD et le forex, expliquées par un chatbot (Sentinel). Posture éducative, conforme UE 2024/2811.',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'MIA Markets — Indicateur de marché conversationnel',
    description:
      'Comprenez le marché — sans qu’on vous dise quoi faire. Lectures algorithmiques + chatbot Sentinel.',
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
  return (
    <html lang={locale} className={inter.variable} suppressHydrationWarning>
      <body className="min-h-screen bg-background font-sans antialiased">
        <SkipLink />
        <PlausibleScript />
        <JsonLd data={softwareApplicationLd} />
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <NextIntlClientProvider messages={messages}>
            <TooltipProvider delayDuration={200}>
              <ChatProvider>
                <Nav />
                <main id="main" className="min-h-[calc(100vh-160px)]">
                  {children}
                </main>
                <Footer />
                <ChatPanel />
                <CookieBanner />
              </ChatProvider>
            </TooltipProvider>
          </NextIntlClientProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
