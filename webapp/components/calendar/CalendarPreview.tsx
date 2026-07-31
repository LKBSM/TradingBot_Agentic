'use client';

import * as React from 'react';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { ChevronRight } from 'lucide-react';
import { useLocalizedHref } from '@/lib/i18n/href';
import { useCalendar } from '@/lib/calendar/useCalendar';
import { filterEvents, hmInZone, splitPastUpcoming } from '@/lib/calendar/grouping';
import { parseUtc } from '@/lib/time/localTime';
import type { CalendarResponse } from '@/types/calendar';
import './calendar.css';

const ALL_SOURCES = new Set([
  'bls', 'bea', 'census', 'federal_reserve', 'eurostat', 'ecb', 'forexfactory',
]);
const ALL_MARKETS = new Set(['XAUUSD', 'EURUSD']);
const ALL_PERIODICITIES = new Set(['monthly', 'quarterly', 'eight_per_year']);

/**
 * Compact dashboard preview of the next few scheduled publications — the SAME
 * source and helpers as the full /actualites page (no logic duplication), just
 * capped and linking through. Announces moments only.
 *
 * NOTE (NW-1): this component is ready but is intentionally NOT yet placed on
 * /app. Swapping the existing events module for it is deferred to the
 * official-source integration mission, so the preview shows real publications
 * instead of emptying /app while only the prototype/stub source exists.
 */
export function CalendarPreview({
  limit = 3,
  data: injectedData,
  now: injectedNow,
}: {
  limit?: number;
  data?: CalendarResponse | null;
  now?: Date;
}) {
  const t = useTranslations('calendar');
  const lh = useLocalizedHref();
  const hook = useCalendar();
  const data = injectedData !== undefined ? injectedData : hook.data;
  const now = React.useMemo(() => injectedNow ?? new Date(), [injectedNow]);

  const upcoming = data
    ? splitPastUpcoming(
        filterEvents(data.events, ALL_SOURCES, ALL_MARKETS, ALL_PERIODICITIES),
        now,
      ).upcoming.slice(0, limit)
    : [];

  return (
    <section className="calprev">
      <div className="calprev-h">
        <span>{t('preview.title')}</span>
        <Link className="calprev-link" href={lh('/actualites')}>
          {t('preview.seeAll')}
          <ChevronRight width={12} height={12} aria-hidden />
        </Link>
      </div>
      {upcoming.length === 0 ? (
        <p className="calprev-empty">{t('preview.empty')}</p>
      ) : (
        <ul className="calprev-list">
          {upcoming.map((ev) => {
            const when = parseUtc(ev.scheduled_at);
            return (
              <li key={ev.event_id} className="calprev-row">
                <span className="calprev-time mono">
                  {when ? hmInZone(when, ev.source_timezone ?? undefined) : '—'}
                </span>
                <span className="calprev-ev">{ev.event}</span>
                {ev.periodicity && (
                  <span className="calprev-per">
                    {t(`periodicity.${ev.periodicity}`)}
                  </span>
                )}
                {/* Deep-link to the OFFICIAL event detail (loads by id, REC 1). */}
                <Link
                  className="calprev-more"
                  href={lh(`/actualites/${encodeURIComponent(ev.event_id)}`)}
                  aria-label={`${t('seeMore')} — ${ev.event}`}
                >
                  {t('seeMore')}
                  <ChevronRight width={12} height={12} aria-hidden />
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
