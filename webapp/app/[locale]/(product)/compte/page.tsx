import type { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { AccountPanel } from '@/components/auth/AccountPanel';
import { SubscriptionGate } from '@/components/access/SubscriptionGate';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'pages' });
  return {
    title: t('compte.meta.title'),
    description: t('compte.meta.description'),
    robots: { index: false, follow: false },
  };
}

export default function AccountPage() {
  // Gate the private account page like the other product surfaces (AUTH-05):
  // under the beta/payment gate an anonymous visitor is bounced to login rather
  // than left on a spinner. requireSubscription=false — an authenticated but
  // UNSUBSCRIBED account MUST reach its account page (PAY-1: the account page +
  // subscribe invitation are exactly what it may see). No paywall here.
  // AccountPanel still guards its own null/probe states.
  return (
    <SubscriptionGate requireSubscription={false}>
      <AccountPanel />
    </SubscriptionGate>
  );
}
