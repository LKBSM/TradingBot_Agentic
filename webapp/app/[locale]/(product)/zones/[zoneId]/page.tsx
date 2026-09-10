import type { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { ZoneDetail } from '@/components/zones/ZoneDetail';
import { SubscriptionGate } from '@/components/access/SubscriptionGate';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'zones' });
  return {
    title: t('detail.meta.title'),
    description: t('detail.meta.description'),
  };
}

/**
 * VZ-4 — `/zones/<engine zone id>`: the detailed sheet of ONE detected zone,
 * reached from the compact /zones card and from the /app structure list (both
 * paths land here). It sits in the same product shell as /zones, so the rail and
 * the single M.I.A column are the ones the rest of the product uses.
 *
 * The id in the URL is the zone's REAL engine id — it is looked up in the
 * reading, never matched by price or time (mission §4 id lock). An id the
 * current reading no longer carries renders an honest "no longer detected"
 * state, never a look-alike zone.
 */
export default async function ZoneDetailPage({
  params,
}: {
  params: Promise<{ locale: string; zoneId: string }>;
}) {
  const { zoneId } = await params;
  return (
    <SubscriptionGate>
      <ZoneDetail zoneId={decodeURIComponent(zoneId)} />
    </SubscriptionGate>
  );
}
