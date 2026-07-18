import type { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { ZonesWorkspace } from '@/components/zones/ZonesWorkspace';
import { SubscriptionGate } from '@/components/access/SubscriptionGate';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'pages' });
  return {
    title: t('zones.meta.title'),
    description: t('zones.meta.description'),
  };
}

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
