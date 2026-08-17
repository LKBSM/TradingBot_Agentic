import type { Metadata } from 'next';
import { Suspense } from 'react';
import { useTranslations } from 'next-intl';
import { getTranslations } from 'next-intl/server';
import { SubscriptionPanel } from '@/components/billing/SubscriptionPanel';
import { SubscriptionGate } from '@/components/access/SubscriptionGate';
import { AuthBrandHeader } from '@/components/auth/AuthBrandHeader';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'pages' });
  return {
    title: t('abonnement.meta.title'),
    description: t('abonnement.meta.description'),
    robots: { index: false, follow: false },
  };
}

export default function SubscriptionPage() {
  const t = useTranslations('pages');
  return (
    // Gate the subscription page (AUTH-05): a logged-out visitor under the gate
    // is redirected to login instead of stranded on the panel's own loader.
    // requireSubscription=false — this is where an UNSUBSCRIBED account subscribes,
    // so it must never be paywalled (PAY-1).
    <SubscriptionGate requireSubscription={false}>
      <div className="container-prose py-12 sm:py-16">
        <AuthBrandHeader />
        {/* SubscriptionPanel reads search params (Checkout return status) — wrap
            in Suspense so the static shell can render while it hydrates. */}
        <Suspense
          fallback={
            <p className="text-sm text-muted-foreground">
              {t('abonnement.loading')}
            </p>
          }
        >
          <SubscriptionPanel />
        </Suspense>
      </div>
    </SubscriptionGate>
  );
}
