'use client';

import * as React from 'react';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { ChevronRight } from 'lucide-react';
import { useLocalizedHref } from '@/lib/i18n/href';
import { useMultiFilter } from '@/lib/market-reading/use-multi-filter';
import { FilterChipGroup } from '@/components/app/FilterChipGroup';
import { useCalendar } from '@/lib/calendar/useCalendar';
import {
  countdown,
  filterEvents,
  groupEventsByDay,
  hmInZone,
  longDayLabel,
  splitPastUpcoming,
} from '@/lib/calendar/grouping';
import { parseUtc, utcOffsetLabel } from '@/lib/time/localTime';
import type { CalendarEvent, CalendarImpact, CalendarResponse } from '@/types/calendar';
import '@/components/app/ui2c.css'; // reuse the shared .fchip filter chips
import './calendar.css';

const IMPACTS: readonly CalendarImpact[] = ['high', 'medium', 'low'];
const MARKETS = ['XAUUSD', 'EURUSD'] as const;
const POLL_MS = 60_000;

function capitalize(s: string): string {
  return s.length ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

/** Last IANA segment as a human label: "America/New_York" → "New York". */
function tzCity(iana: string | null): string | null {
  if (!iana) return null;
  const seg = iana.split('/').pop() ?? iana;
  return seg.replace(/_/g, ' ');
}

/**
 * /actualites — the scheduled-volatility calendar (NW-1, LIST view only).
 * Announces MOMENTS, never DIRECTIONS. Chronological only — no amplitude sort,
 * no colour ranking. Fields the source does not provide render as absent.
 *
 * `data`/`now` are injectable for tests; live it reads GET /api/calendar (whose
 * source defaults to the official stub until official feeds are wired).
 */
export function CalendarWorkspace({
  locale,
  data: injectedData,
  now: injectedNow,
}: {
  locale: string;
  data?: CalendarResponse | null;
  now?: Date;
}) {
  const t = useTranslations('calendar');
  const hook = useCalendar({ pollMs: POLL_MS });
  const data = injectedData !== undefined ? injectedData : hook.data;
  const isLoading = injectedData !== undefined ? false : hook.isLoading;
  const error = injectedData !== undefined ? null : hook.error;

  const impactFilter = useMultiFilter<CalendarImpact>(IMPACTS);
  const marketFilter = useMultiFilter<(typeof MARKETS)[number]>(MARKETS);
  const [showPast, setShowPast] = React.useState(false);

  // A single "now" for the whole render (stable countdowns). Tests inject it.
  const now = React.useMemo(() => injectedNow ?? new Date(), [injectedNow]);
  const offsetLabel = utcOffsetLabel();

  const marketName = React.useCallback(
    (m: string) => t(`market.${m}` as 'market.XAUUSD'),
    [t],
  );

  const impactOptions = IMPACTS.map((v) => ({ value: v, label: t(`impact.${v}`) }));
  const marketOptions = MARKETS.map((v) => ({ value: v, label: marketName(v) }));

  const marketsHeader = MARKETS.map(marketName).join(' · ');

  return (
    <div className="cal-page">
      <div className="cal-head">
        <h1>{t('title')}</h1>
        <span className="cal-badge">{t('badge')}</span>
        <span className="cal-markets mono">{marketsHeader}</span>
      </div>

      <div className="cal-intro">
        <b>{t('intro.lead')}</b> {t('intro.rest')}
      </div>

      <div className="cal-filtbar">
        <div>
          <FilterChipGroup
            label={t('filters.impactLabel')}
            options={impactOptions}
            filter={impactFilter}
            resetLabel={t('filters.reset')}
          />
        </div>
        <div>
          <FilterChipGroup
            label={t('filters.marketLabel')}
            options={marketOptions}
            filter={marketFilter}
            resetLabel={t('filters.reset')}
          />
        </div>
        <span className="cal-spacer" />
        <button
          type="button"
          className={`cal-past${showPast ? ' on' : ''}`}
          aria-pressed={showPast}
          onClick={() => setShowPast((v) => !v)}
        >
          {t('filters.past')}
        </button>
      </div>

      <Body
        t={t}
        locale={locale}
        now={now}
        data={data}
        isLoading={isLoading}
        error={error}
        noneSelected={impactFilter.noneSelected || marketFilter.noneSelected}
        impacts={impactFilter.selected}
        markets={marketFilter.selected}
        showPast={showPast}
        offsetLabel={offsetLabel}
        marketName={marketName}
      />

      <div className="cal-nono" role="note">
        <div>
          <div className="t">{t('nono.title')}</div>
          <div className="b">{t('nono.body')}</div>
        </div>
      </div>
    </div>
  );
}

function Body(props: {
  t: ReturnType<typeof useTranslations>;
  locale: string;
  now: Date;
  data: CalendarResponse | null | undefined;
  isLoading: boolean;
  error: Error | null;
  noneSelected: boolean;
  impacts: ReadonlySet<string>;
  markets: ReadonlySet<string>;
  showPast: boolean;
  offsetLabel: string;
  marketName: (m: string) => string;
}) {
  const {
    t, locale, now, data, isLoading, error, noneSelected,
    impacts, markets, showPast, offsetLabel, marketName,
  } = props;

  if (isLoading) return <div className="cal-status">{t('loading')}</div>;
  if (error) return <div className="cal-status">{t('error')}</div>;
  if (!data) return <div className="cal-status">{t('error')}</div>;

  if (noneSelected) {
    return <div className="cal-empty">{t('empty.noSelection')}</div>;
  }

  const filtered = filterEvents(data.events, impacts, markets);
  const { past, upcoming } = splitPastUpcoming(filtered, now);
  const shown = showPast ? past : upcoming;
  const groups = groupEventsByDay(shown, now);

  const sourceIsOfficialStub =
    data.coverage.source === 'official' && data.events.length === 0;

  if (groups.length === 0) {
    if (sourceIsOfficialStub) {
      return <div className="cal-empty">{t('empty.noSource')}</div>;
    }
    return <div className="cal-empty">{t('empty.noEvents')}</div>;
  }

  return (
    <>
      {groups.map((g) => {
        const prefix = g.isToday
          ? t('day.today')
          : g.isTomorrow
            ? t('day.tomorrow')
            : null;
        const long = longDayLabel(g.date, locale);
        const label = prefix ? `${prefix} · ${long}` : capitalize(long);
        return (
          <React.Fragment key={g.key}>
            <div className="cal-daysep">
              {label}
              <span className="cnt">
                {t('day.count', { count: g.events.length })}
              </span>
            </div>
            {g.events.map((ev) => (
              <Row
                key={ev.event_id}
                ev={ev}
                now={now}
                t={t}
                past={props.showPast}
                offsetLabel={offsetLabel}
                marketName={marketName}
              />
            ))}
          </React.Fragment>
        );
      })}

      {data.coverage.partial && data.coverage.feed_end && (
        <div className="cal-coverage">
          {t('coverage.partial', {
            date: longDayLabel(parseUtc(data.coverage.feed_end) ?? now, locale),
          })}
        </div>
      )}
    </>
  );
}

function Row({
  ev,
  now,
  t,
  past,
  offsetLabel,
  marketName,
}: {
  ev: CalendarEvent;
  now: Date;
  t: ReturnType<typeof useTranslations>;
  past: boolean;
  offsetLabel: string;
  marketName: (m: string) => string;
}) {
  const lh = useLocalizedHref();
  const when = parseUtc(ev.scheduled_at);
  const cd = when ? countdown(now.getTime(), when.getTime()) : null;
  const city = tzCity(ev.source_timezone);
  const marketTime = when ? hmInZone(when, ev.source_timezone ?? undefined) : '—';
  const localTime = when ? hmInZone(when) : '—';
  const affects = ev.markets.map(marketName).join(', ');

  return (
    <div className={`cal-row${past ? ' past' : ''}`}>
      <div className="cal-clockcol">
        <div className="cal-clock">
          {marketTime}
          {city && <span className="tz"> {city}</span>}
        </div>
        {cd && <div className="cal-cd">{formatCountdown(cd, t)}</div>}
        <div className="cal-local">
          {t('localTime', { time: localTime, offset: offsetLabel })}
        </div>
      </div>

      <div className="cal-mid">
        <div className="cal-ev">{ev.event}</div>
        <div className="cal-meta">
          <span className="cal-mkchip">{ev.currency}</span>
          <span className="sep">·</span>
          <span>{t('affects', { markets: affects })}</span>
          {ev.revised && (
            <>
              <span className="sep">·</span>
              <span>{t('revisedBadge')}</span>
            </>
          )}
        </div>
        <div className="cal-prov">
          {t('provenance.source', { source: sourceLabel(ev.source, t) })}
          {' · '}
          {ev.organism ? (
            t('provenance.organism', { organism: ev.organism })
          ) : (
            <span className="missing">{t('provenance.organismMissing')}</span>
          )}
        </div>
      </div>

      <span className={`cal-impact ${ev.impact}`}>{t(`impact.${ev.impact}`)}</span>

      <div className="cal-ampl">
        <div className="k">{t('amplitude.k')}</div>
        <div className="v">{t('amplitude.pending')}</div>
      </div>

      <Link
        className="cal-more"
        href={lh(`/actualites/${encodeURIComponent(ev.event_id)}`)}
        aria-label={`${t('seeMore')} — ${ev.event}`}
      >
        {t('seeMore')}
        <ChevronRight width={13} height={13} aria-hidden />
      </Link>
    </div>
  );
}

function sourceLabel(source: string, t: ReturnType<typeof useTranslations>): string {
  if (source === 'official') return t('provenance.sourceName.official');
  if (source === 'forexfactory') return t('provenance.sourceName.forexfactory');
  return source;
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
