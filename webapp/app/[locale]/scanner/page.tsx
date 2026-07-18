import type { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { ScannerWorkspace } from '@/components/scanner/ScannerWorkspace';
import { SubscriptionGate } from '@/components/access/SubscriptionGate';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'pages' });
  return {
    title: t('scanner.meta.title'),
    description: t('scanner.meta.description'),
  };
}

/**
 * Scanner page — the user composes present-tense structural conditions and the
 * tool shows where they are met right now across XAU/USD and EUR/USD on M15/H1/H4.
 * Descriptive and read-only: it reuses readings the detection engine already
 * produced and never recommends action.
 */
export default async function ScannerPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'pages' });
  // The multi-market scanner is a paid feature: require full access (subscriber
  // or owner). Visitors → login; free accounts → a clean upsell paywall. Open
  // while the gate is OFF (testing phase).
  return (
    <SubscriptionGate
      requireFullAccess
      paywallTitle={t('scanner.paywallTitle')}
      paywallDescription={t('scanner.paywallDescription')}
    >
      <div className="container-wide py-8">
        <ScannerWorkspace locale={locale} />
      </div>
    </SubscriptionGate>
  );
}
