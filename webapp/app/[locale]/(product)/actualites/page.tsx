import type { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { CalendarMonthView } from '@/components/calendar/CalendarMonthView';
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
 * /actualites — "Actualités programmées", the scheduled-volatility calendar
 * (NW-1, list view). Announces MOMENTS (publication times, source, organism,
 * attached market), never DIRECTIONS. Read-only; the active data source is
 * env-driven server-side and defaults to the official stub until the official
 * feeds (BLS/BEA/Census/Fed, Eurostat/ECB) are integrated in a follow-up.
 */
export default async function ActualitesPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  return (
    <SubscriptionGate>
      <CalendarMonthView locale={locale} />
    </SubscriptionGate>
  );
}
