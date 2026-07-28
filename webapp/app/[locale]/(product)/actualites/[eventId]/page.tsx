import type { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { CalendarEventDetail } from '@/components/calendar/CalendarEventDetail';
import { SubscriptionGate } from '@/components/access/SubscriptionGate';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'calendar' });
  return {
    title: t('meta.title'),
    description: t('meta.description'),
  };
}

/**
 * /actualites/[eventId] — the page for one scheduled publication, deep-linked
 * from the list « En savoir plus ». Honest and minimal: it shows the real fields
 * the source provides (times, attached market, attributed impact, published
 * figures when present, provenance) and marks the engine-measured history as
 * « mesures à venir » (NW-2). Announces a moment, never a direction.
 */
export default async function ActualiteDetailPage({
  params,
}: {
  params: Promise<{ locale: string; eventId: string }>;
}) {
  const { locale, eventId } = await params;
  return (
    <SubscriptionGate>
      <CalendarEventDetail eventId={decodeURIComponent(eventId)} locale={locale} />
    </SubscriptionGate>
  );
}
