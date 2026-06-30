import type { Metadata } from 'next';
import { ZonesWorkspace } from '@/components/zones/ZonesWorkspace';
import { SubscriptionGate } from '@/components/access/SubscriptionGate';

export const metadata: Metadata = {
  title: 'Zones',
  description:
    'Le cycle de vie de chaque zone détectée (Order Block, Fair Value Gap) — formation, tests, mitigation, comblement. Lecture descriptive et read-only : aucune prévision.',
};

/**
 * Zones page — the life of every detected OB/FVG zone for a chosen combo:
 * formation → tests → mitigation/fill, as a lifecycle timeline plus one factual
 * sentence. Strictly descriptive and read-only: it reuses the reading the
 * detection engine already produced (same source as /app) and never recommends
 * action. "Analyser" focuses the zone on the chart; "Masquer" hides it from the
 * chart — both display-only, by the zone's real engine id.
 */
export default async function ZonesPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  return (
    <SubscriptionGate>
      <div className="container-wide py-8">
        <ZonesWorkspace locale={locale} />
      </div>
    </SubscriptionGate>
  );
}
