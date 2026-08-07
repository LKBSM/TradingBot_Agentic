import type { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { ConversationalScanner } from '@/components/scanner/conversational/ConversationalScanner';
import { ScannerModeToggle } from '@/components/scanner/ScannerModeToggle';
import { SubscriptionGate } from '@/components/access/SubscriptionGate';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'scannerChat' });
  return {
    title: t('meta.title'),
    description: t('meta.description'),
  };
}

/**
 * SC-2 — conversational scanner ("Décrire"). Describe a strategy in natural
 * language (typed or dictated); M.I.A translates it toward the closed palette,
 * shows what she understood / assumed / could not translate, and nothing runs
 * before the user validates. An additional entry to the same read-only scan —
 * never a replacement for the manual palette.
 */
export default async function ConversationalScannerPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'pages' });
  return (
    <SubscriptionGate
      requireFullAccess
      paywallTitle={t('scanner.paywallTitle')}
      paywallDescription={t('scanner.paywallDescription')}
    >
      <div className="space-y-4">
        <ScannerModeToggle locale={locale} active="describe" />
        <ConversationalScanner locale={locale} />
      </div>
    </SubscriptionGate>
  );
}
