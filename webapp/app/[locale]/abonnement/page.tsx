import type { Metadata } from 'next';
import { Suspense } from 'react';
import { SubscriptionPanel } from '@/components/billing/SubscriptionPanel';

export const metadata: Metadata = {
  title: 'Abonnement',
  description: 'Gérer votre abonnement MIA Markets.',
  robots: { index: false, follow: false },
};

export default function SubscriptionPage() {
  return (
    <div className="container-prose py-12 sm:py-16">
      {/* SubscriptionPanel reads search params (Checkout return status) — wrap
          in Suspense so the static shell can render while it hydrates. */}
      <Suspense
        fallback={<p className="text-sm text-muted-foreground">Chargement…</p>}
      >
        <SubscriptionPanel />
      </Suspense>
    </div>
  );
}
