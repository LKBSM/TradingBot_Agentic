'use client';

import * as React from 'react';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { ChevronLeft } from 'lucide-react';
import { useLocalizedHref } from '@/lib/i18n/href';
import { useCalendar } from '@/lib/calendar/useCalendar';
import { countdown, hmInZone } from '@/lib/calendar/grouping';
import { parseUtc, utcOffsetLabel, formatLocalDayLong } from '@/lib/time/localTime';
import type { CalendarEvent, CalendarResponse } from '@/types/calendar';
import './calendar.css';

// Wide window so a deep-linked event is found even if the list scrolled.
const LOOKAHEAD_DAYS = 30;
const LOOKBACK_DAYS = 30;

/** Render a published value AS PUBLISHED — no conversion, no re-rounding. */
function asPublished(n: number | null): string {
  return n == null ? '—' : String(n);
}

function tzCity(iana: string | null): string | null {
  if (!iana) return null;
  const seg = iana.split('/').pop() ?? iana;
  return seg.replace(/_/g, ' ');
}

/**
 * Per-event detail page (deep-linked from the list « En savoir plus »). Honest
 * and MINIMAL: it shows only the fields the source actually provides — market +
 * local time, attached markets, periodicity, published figures with their
 * revision history (initial value, current value, revision date), source +
 * organism + unit + licence. NO consensus (no organism publishes one), NO impact
 * ranking. It announces a MOMENT, never a direction.
 */
export function CalendarEventDetail({
  eventId,
  locale,
  data: injectedData,
  now: injectedNow,
}: {
  eventId: string;
  locale: string;
  data?: CalendarResponse | null;
  now?: Date;
}) {
  const t = useTranslations('calendar');
  const lh = useLocalizedHref();
  const hook = useCalendar({ lookaheadDays: LOOKAHEAD_DAYS, lookbackDays: LOOKBACK_DAYS });
  const data = injectedData !== undefined ? injectedData : hook.data;
  const isLoading = injectedData !== undefined ? false : hook.isLoading;
  const error = injectedData !== undefined ? null : hook.error;
  const now = React.useMemo(() => injectedNow ?? new Date(), [injectedNow]);

  // Match the full id ("<source>:<ref>") or the bare ref — the App news module
  // deep-links with the raw pipeline ref (no source prefix).
  const ev =
    data?.events.find(
      (e) => e.event_id === eventId || e.event_id.split(':').pop() === eventId,
    ) ?? null;

  const attribution = ev
    ? data?.attribution.find((a) => a.source === ev.source) ?? null
    : null;

  return (
    <div className="cal-page cald">
      <Link className="cald-back" href={lh('/actualites')}>
        <ChevronLeft width={14} height={14} aria-hidden />
        {t('detail.back')}
      </Link>

      {isLoading ? (
        <div className="cal-status">{t('loading')}</div>
      ) : error ? (
        <div className="cal-status">{t('error')}</div>
      ) : !ev ? (
        <div className="cal-empty">
          <strong>{t('detail.notFoundTitle')}</strong>
          <br />
          {t('detail.notFound')}
        </div>
      ) : (
        <Detail ev={ev} attribution={attribution} now={now} locale={locale} t={t} />
      )}
    </div>
  );
}

function fmtCountdown(
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

/**
 * Engine-measured history card. Product invariant: no data, no element. It
 * renders NOTHING until a measure exists for this event — the page never shows
 * an empty tile, nor a promise of future content. The component stays in place,
 * ready to display the moment the measure model is populated.
 */
function EngineMeasuresCard({
  t,
}: {
  t: ReturnType<typeof useTranslations>;
}) {
  // No engine-measure model is attached to the event yet; while there is none,
  // the card is not rendered (Section 3 populates and renders it).
  const measures: readonly unknown[] = [];
  if (measures.length === 0) return null;
  return (
    <div className="cald-card">
      <div className="cald-card-h">
        <h3>{t('detail.measuresTitle')}</h3>
        <span className="cald-badge">{t('detail.measuresBadge')}</span>
      </div>
    </div>
  );
}

function Detail({
  ev,
  attribution,
  now,
  locale,
  t,
}: {
  ev: CalendarEvent;
  attribution: { organism: string; license_label: string; policy_url: string } | null;
  now: Date;
  locale: string;
  t: ReturnType<typeof useTranslations>;
}) {
  const when = parseUtc(ev.scheduled_at);
  const cd = when ? countdown(now.getTime(), when.getTime()) : null;
  const city = tzCity(ev.source_timezone);
  const marketTime = when ? hmInZone(when, ev.source_timezone ?? undefined) : '—';
  const localTime = when ? hmInZone(when) : '—';
  const dayLabel = when ? formatLocalDayLong(when, locale) : '—';
  const affects = ev.markets.map((m) => t(`market.${m}` as 'market.XAUUSD')).join(', ');
  const revisedDate = ev.revised_at ? parseUtc(ev.revised_at) : null;
  const revisedDateLabel = revisedDate ? revisedDate.toLocaleDateString(locale) : '—';

  const nonoItems = Object.values(
    t.raw('detail.nono.items') as Record<string, string>,
  );

  return (
    <>
      <div className="cald-head">
        <div className="cald-flag mono">{ev.currency}</div>
        <div className="cald-headmain">
          <h1>{ev.event}</h1>
          <div className="cald-sub">
            <span>{t('affects', { markets: affects })}</span>
            {ev.periodicity && (
              <>
                <span className="sep">·</span>
                <span>{t(`periodicity.${ev.periodicity}`)}</span>
              </>
            )}
            {!ev.time_confirmed && (
              <>
                <span className="sep">·</span>
                <span>{t('timeUnconfirmed')}</span>
              </>
            )}
            {ev.revised && (
              <>
                <span className="sep">·</span>
                <span>{t('revisedBadge')}</span>
              </>
            )}
          </div>
          <div className="cald-times mono">
            {dayLabel} · {marketTime}
            {city ? ` ${city}` : ''} · {t('localTime', { time: localTime, offset: utcOffsetLabel() })}
          </div>
          <div className="cald-prov">
            {ev.organism ? (
              t('provenance.organism', { organism: ev.organism })
            ) : (
              <span className="missing">{t('provenance.organismMissing')}</span>
            )}
            {' · '}
            {ev.value_unit ? ev.value_unit : (
              <span className="missing">{t('detail.unitMissing')}</span>
            )}
          </div>
        </div>
        {cd && (
          <div className="cald-cd">
            <div className="k">{t('detail.countdownLabel')}</div>
            <div className="v mono">{fmtCountdown(cd, t)}</div>
          </div>
        )}
      </div>

      {/* Chiffres publiés — valeur de l'indicateur, telle que publiée, jamais un prix */}
      <div className="cald-card">
        <div className="cald-card-h">
          <h3>{t('detail.publishedFiguresTitle')}</h3>
          <span className="cald-badge">{t('detail.publishedFiguresBadge')}</span>
        </div>
        <div className="cald-figs">
          <div className="cald-fig">
            <div className="k">{t('detail.actualLabel')}</div>
            <div className="v mono">{asPublished(ev.actual)}</div>
            {ev.actual == null && <div className="n">{t('detail.actualPending')}</div>}
          </div>
          <div className="cald-fig">
            <div className="k">{t('detail.previousLabel')}</div>
            <div className="v mono">{asPublished(ev.previous)}</div>
          </div>
        </div>

        {/* Révisions — la valeur avant, la valeur après, la date. Aucune qualification. */}
        <div className="cald-rev">
          {ev.revised ? (
            <p className="cald-rev-line">
              {t('detail.revisedFromTo', {
                initial: asPublished(ev.actual_initial),
                current: asPublished(ev.actual),
                date: revisedDateLabel,
              })}
            </p>
          ) : ev.actual != null ? (
            <p className="cald-rev-line">{t('detail.notRevised')}</p>
          ) : null}
        </div>

        <p className="cald-note">{t('detail.publishedFiguresNote')}</p>
      </div>

      <EngineMeasuresCard t={t} />

      {/* Attribution — condition de licence de la source */}
      {attribution && (
        <div className="cal-attrib" role="contentinfo">
          <div className="t">{t('attribution.title')}</div>
          <ul>
            <li>
              <span className="org">{attribution.organism}</span>
              <span className="lic">{attribution.license_label}</span>
              <a href={attribution.policy_url} target="_blank" rel="noreferrer noopener">
                {t('attribution.policyLink')}
              </a>
            </li>
          </ul>
        </div>
      )}

      <div className="cal-nono" role="note">
        <div>
          <div className="t">{t('detail.nono.title')}</div>
          <div className="b">
            <ul>
              {nonoItems.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </>
  );
}
