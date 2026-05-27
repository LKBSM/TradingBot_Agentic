import type { Metadata, Viewport } from 'next';
import { NextIntlClientProvider } from 'next-intl';
import { getMessages } from 'next-intl/server';
import { notFound } from 'next/navigation';
import '../globals.css';
import { Nav } from '@/components/Nav';
import { Footer } from '@/components/Footer';
import { ChatPanel } from '@/components/chat/ChatPanel';
import { ChatProvider } from '@/components/chat/ChatProvider';
import { ThemeProvider } from '@/components/theme-provider';
import { TooltipProvider } from '@/components/ui/tooltip';
import { SUPPORTED_LOCALES } from '../../i18n';

export const metadata: Metadata = {
  title: {
    default: 'M.I.A. Markets — Indicateur de marché conversationnel',
    template: '%s · M.I.A. Markets',
  },
  description:
    "M.I.A. Markets (Multi-asset Intelligence Assistant) — Lecture de marché XAU/USD et FX, expliquée par un quant. Analyses contextuelles sourcées, posture éducative, conforme UE 2024/2811.",
  applicationName: 'M.I.A. Markets',
  authors: [{ name: 'M.I.A. Markets' }],
  robots: { index: true, follow: true },
  // PWA install: iOS Safari reads these to badge the home-screen icon and
  // hide the browser chrome when launched standalone.
  appleWebApp: {
    capable: true,
    title: 'M.I.A.',
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
    <html lang={locale} suppressHydrationWarning>
      <body className="min-h-screen bg-background font-sans antialiased">
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
                <main className="min-h-[calc(100vh-160px)]">{children}</main>
                <Footer />
                <ChatPanel />
              </ChatProvider>
            </TooltipProvider>
          </NextIntlClientProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
