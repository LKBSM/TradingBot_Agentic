import type { Metadata } from 'next';
import { Suspense } from 'react';
import { useTranslations } from 'next-intl';
import { getTranslations } from 'next-intl/server';
import { SubscriptionPanel } from '@/components/billing/SubscriptionPanel';
import { SubscriptionGate } from '@/components/access/SubscriptionGate';

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
    <SubscriptionGate>
      <div className="container-prose py-12 sm:py-16">
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
