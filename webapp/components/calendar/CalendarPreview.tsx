'use client';

import * as React from 'react';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { ChevronRight } from 'lucide-react';
import { useLocalizedHref } from '@/lib/i18n/href';
import { useCalendar } from '@/lib/calendar/useCalendar';
import { OFFICIAL_SOURCES } from '@/lib/calendar/officialSources';
import {
  countdown,
  filterEvents,
  hmInZone,
  splitPastUpcoming,
} from '@/lib/calendar/grouping';
import { parseUtc } from '@/lib/time/localTime';
import type { CalendarResponse } from '@/types/calendar';
import './calendar.css';

// Official issuing organisms only — the explicit production whitelist (CAL-1).
const ALL_SOURCES = new Set<string>(OFFICIAL_SOURCES);
const ALL_MARKETS = new Set(['XAUUSD', 'EURUSD']);
const ALL_PERIODICITIES = new Set(['monthly', 'quarterly', 'eight_per_year']);

// No cap by default: the dashboard shows the FULL forward horizon in a scrollable
// list (mission NW-3 §2C). `limit` stays injectable but defaults to "show all".
const NO_CAP = Number.POSITIVE_INFINITY;

/**
 * Dashboard preview of ALL upcoming scheduled publications — the SAME source and
 * helpers as the full /actualites page (no logic duplication). The list is
 * scrollable and shows every forward release over the fetched horizon (30 days),
 * each row carrying its countdown, local time, name, organism, periodicity, the
 * attached markets and a deep link to the official detail. Announces moments only.
 */
export function CalendarPreview({
  limit = NO_CAP,
  data: injectedData,
  now: injectedNow,
}: {
  limit?: number;
  data?: CalendarResponse | null;
  now?: Date;
}) {
  const t = useTranslations('calendar');
  const lh = useLocalizedHref();
  // A volatility calendar is useful across the coming weeks, not just 7 days —
  // request the full forward horizon so every upcoming release is listed.
  const hook = useCalendar({ lookaheadDays: 30 });
  const data = injectedData !== undefined ? injectedData : hook.data;
  // Tests inject data (⇒ resolved); live, the waiting/error states must be
  // distinct from an empty result so the preview never claims "nothing upcoming"
  // before the data has loaded (CAL-1).
  const isLoading = injectedData !== undefined ? false : hook.isLoading;
  const error = injectedData !== undefined ? null : hook.error;
  const now = React.useMemo(() => injectedNow ?? new Date(), [injectedNow]);

  const marketName = React.useCallback(
    (m: string) => t(`market.${m}` as 'market.XAUUSD'),
    [t],
  );

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
      {isLoading ? (
        <p className="calprev-empty calprev-status" role="status">{t('loading')}</p>
      ) : error ? (
        <p className="calprev-empty calprev-status">{t('error')}</p>
      ) : upcoming.length === 0 ? (
        <p className="calprev-empty">{t('preview.empty')}</p>
      ) : (
        <ul className="calprev-list">
          {upcoming.map((ev) => {
            const when = parseUtc(ev.scheduled_at);
            const cd = when ? countdown(now.getTime(), when.getTime()) : null;
            const localTime = when ? hmInZone(when) : '—';
            const affects = ev.markets.map(marketName).join(', ');
            return (
              <li key={ev.event_id} className="calprev-row">
                <div className="calprev-when">
                  {cd && (
                    <span className="calprev-cd">{formatCountdown(cd, t)}</span>
                  )}
                  <span className="calprev-time mono">{localTime}</span>
                  {!ev.time_confirmed && (
                    <span className="calprev-unconf">{t('timeUnconfirmed')}</span>
                  )}
                </div>
                <div className="calprev-mid">
                  <div className="calprev-ev">{ev.event}</div>
                  <div className="calprev-meta">
                    <span className="calprev-prov">
                      {ev.organism
                        ? t('provenance.organism', { organism: ev.organism })
                        : t('provenance.organismMissing')}
                    </span>
                    {ev.periodicity && (
                      <>
                        <span className="calprev-sep">·</span>
                        <span>{t(`periodicity.${ev.periodicity}`)}</span>
                      </>
                    )}
                    {ev.markets.length > 0 && (
                      <>
                        <span className="calprev-sep">·</span>
                        <span>{t('affects', { markets: affects })}</span>
                      </>
                    )}
                  </div>
                </div>
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

function formatCountdown(
  cd: ReturnType<typeof countdown>,
  t: ReturnType<typeof useTranslations>,
): string {
  const { past, days, hours, minutes } = cd;
  const mm = String(minutes).padStart(2, '0');
  if (past) {
    if (days >= 1) return t('countdown.agoDays', { days });
    if (hours >= 1) return t('countdown.agoHours', { hours, minutes: mm });
    return t('countdown.agoMinutes', { minutes });
  }
  if (days >= 1) return t('countdown.days', { days });
  if (hours >= 1) return t('countdown.hours', { hours, minutes: mm });
  if (minutes >= 1) return t('countdown.minutes', { minutes });
  return t('countdown.now');
}
