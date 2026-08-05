'use client';

import * as React from 'react';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { ChevronRight } from 'lucide-react';
import { useLocalizedHref } from '@/lib/i18n/href';
import { useMultiFilter } from '@/lib/market-reading/use-multi-filter';
import { FilterChipGroup } from '@/components/app/FilterChipGroup';
import { useCalendar } from '@/lib/calendar/useCalendar';
import { OFFICIAL_SOURCES } from '@/lib/calendar/officialSources';
import {
  countdown,
  filterEvents,
  groupEventsByDay,
  hmInZone,
  latestSuccess,
  longDayLabel,
  splitPastUpcoming,
} from '@/lib/calendar/grouping';
import { parseUtc, utcOffsetLabel } from '@/lib/time/localTime';
import type {
  CalendarEvent,
  CalendarPeriodicity,
  CalendarResponse,
} from '@/types/calendar';
import '@/components/app/ui2c.css'; // reuse the shared .fchip filter chips
import './calendar.css';

// Official issuing organisms only — the explicit production whitelist (CAL-1).
const SOURCES = OFFICIAL_SOURCES;
const MARKETS = ['XAUUSD', 'EURUSD'] as const;
const PERIODICITIES: readonly CalendarPeriodicity[] = [
  'monthly', 'quarterly', 'eight_per_year',
];
const POLL_MS = 60_000;

function capitalize(s: string): string {
  return s.length ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

/** Render a published value AS PUBLISHED — no conversion, no re-rounding. */
function asPublished(n: number | null): string {
  return n == null ? '—' : String(n);
}

/** Last IANA segment as a human label: "America/New_York" → "New York". */
function tzCity(iana: string | null): string | null {
  if (!iana) return null;
  const seg = iana.split('/').pop() ?? iana;
  return seg.replace(/_/g, ' ');
}

/**
 * /actualites — the scheduled-volatility calendar (NW-1b, LIST view only).
 * Announces MOMENTS, never DIRECTIONS. Chronological only — no amplitude sort,
 * no impact ranking (no organism grades its releases), no consensus (no organism
 * publishes one). Filters are factual: issuing organism, market, periodicity.
 * Fields the source does not provide render as absent.
 *
 * `data`/`now` are injectable for tests; live it reads GET /api/calendar (whose
 * source defaults to the official aggregator).
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
  // A volatility calendar is useful across the coming weeks, not just 7 days —
  // request the full forward horizon (API cap) so upcoming releases are visible.
  const hook = useCalendar({ lookaheadDays: 30, lookbackDays: 3, pollMs: POLL_MS });
  const data = injectedData !== undefined ? injectedData : hook.data;
  const isLoading = injectedData !== undefined ? false : hook.isLoading;
  const error = injectedData !== undefined ? null : hook.error;

  const sourceFilter = useMultiFilter<(typeof SOURCES)[number]>(SOURCES);
  const marketFilter = useMultiFilter<(typeof MARKETS)[number]>(MARKETS);
  const periodicityFilter = useMultiFilter<CalendarPeriodicity>(PERIODICITIES);
  const [showPast, setShowPast] = React.useState(false);

  // A single "now" for the whole render (stable countdowns). Tests inject it.
  const now = React.useMemo(() => injectedNow ?? new Date(), [injectedNow]);
  const offsetLabel = utcOffsetLabel();

  const marketName = React.useCallback(
    (m: string) => t(`market.${m}` as 'market.XAUUSD'),
    [t],
  );
  const organismName = React.useCallback(
    (s: string) => t(`organism.${s}` as 'organism.bls'),
    [t],
  );

  const sourceOptions = SOURCES.map((v) => ({ value: v, label: organismName(v) }));
  const marketOptions = MARKETS.map((v) => ({ value: v, label: marketName(v) }));
  const periodicityOptions = PERIODICITIES.map((v) => ({
    value: v, label: t(`periodicity.${v}`),
  }));

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
            label={t('filters.organismLabel')}
            options={sourceOptions}
            filter={sourceFilter}
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
        <div>
          <FilterChipGroup
            label={t('filters.periodicityLabel')}
            options={periodicityOptions}
            filter={periodicityFilter}
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
        noneSelected={
          sourceFilter.noneSelected ||
          marketFilter.noneSelected ||
          periodicityFilter.noneSelected
        }
        sources={sourceFilter.selected}
        markets={marketFilter.selected}
        periodicities={periodicityFilter.selected}
        showPast={showPast}
        offsetLabel={offsetLabel}
        marketName={marketName}
      />

      <Attribution t={t} data={data} locale={locale} />

      <div className="cal-nono" role="note">
        <div>
          <div className="t">{t('nono.title')}</div>
          <div className="b">{t('nono.body')}</div>
          <ul className="cal-nono-list">
            <li>{t('nono.noForecast')}</li>
            <li>{t('nono.noRanking')}</li>
            <li>{t('nono.noUnscheduled')}</li>
          </ul>
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
  sources: ReadonlySet<string>;
  markets: ReadonlySet<string>;
  periodicities: ReadonlySet<string>;
  showPast: boolean;
  offsetLabel: string;
  marketName: (m: string) => string;
}) {
  const {
    t, locale, now, data, isLoading, error, noneSelected,
    sources, markets, periodicities, showPast, offsetLabel, marketName,
  } = props;

  if (isLoading) return <div className="cal-status">{t('loading')}</div>;
  if (error) return <div className="cal-status">{t('error')}</div>;
  if (!data) return <div className="cal-status">{t('error')}</div>;

  if (noneSelected) {
    return <div className="cal-empty">{t('empty.noSelection')}</div>;
  }

  const filtered = filterEvents(data.events, sources, markets, periodicities);
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
        {!ev.time_confirmed && (
          <div className="cal-unconf">{t('timeUnconfirmed')}</div>
        )}
      </div>

      <div className="cal-mid">
        <div className="cal-ev">{ev.event}</div>
        <div className="cal-meta">
          <span className="cal-mkchip">{ev.currency}</span>
          <span className="sep">·</span>
          <span>{t('affects', { markets: affects })}</span>
          {ev.periodicity && (
            <>
              <span className="sep">·</span>
              <span>{t(`periodicity.${ev.periodicity}`)}</span>
            </>
          )}
          {ev.revised && (
            <>
              <span className="sep">·</span>
              <span>{t('revisedBadge')}</span>
            </>
          )}
        </div>
        <div className="cal-prov">
          {ev.organism ? (
            t('provenance.organism', { organism: ev.organism })
          ) : (
            <span className="missing">{t('provenance.organismMissing')}</span>
          )}
        </div>
      </div>

      {/* Value + its explicit state — consistent with the detail (no bare dash).
          Pending shows nothing here: the countdown already states it's upcoming.
          No amplitude slot while no measure exists (no future-content promise). */}
      <div className="cal-val">
        {ev.actual_state === 'published' && (
          <span className="v mono">{asPublished(ev.actual)}</span>
        )}
        {ev.actual_state === 'unfetched' && (
          <span className="s">{t('list.unfetched')}</span>
        )}
        {ev.actual_state === 'unavailable' && (
          <span className="s">{t('list.unavailable')}</span>
        )}
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

/**
 * Licence-required attribution block: names every source that actually produced
 * a served event and links its reuse policy. Also surfaces any source that was
 * NOT refreshed (its stored data is kept, its last-success date is shown).
 */
function Attribution({
  t,
  data,
  locale,
}: {
  t: ReturnType<typeof useTranslations>;
  data: CalendarResponse | null | undefined;
  locale: string;
}) {
  if (!data || data.attribution.length === 0) return null;
  const staleSet = new Set(data.coverage.stale_sources);
  // Overall last successful refresh — shown permanently, so the client sees the
  // PROOF of freshness, not merely the absence of a stale marker (NW-4 ch.5).
  const lastUpdated = latestSuccess(data.coverage.last_success);
  return (
    <div className="cal-attrib" role="contentinfo">
      <div className="t">{t('attribution.title')}</div>
      <p className="i">{t('attribution.intro')}</p>
      {lastUpdated && (
        <p className="lastup">
          {t('attribution.lastUpdated', {
            date: lastUpdated.toLocaleDateString(locale),
          })}
        </p>
      )}
      <ul>
        {data.attribution.map((a) => {
          // Per-organism freshness: the last successful refresh, and a marker when
          // it is stale (older than the threshold or missed the latest cycle). A
          // retrieval delay is never silent.
          const iso = data.coverage.last_success[a.source];
          const d = iso ? parseUtc(iso) : null;
          const when = d ? d.toLocaleDateString(locale) : '—';
          const isStale = staleSet.has(a.source);
          return (
            <li key={a.source}>
              <span className="org">{a.organism}</span>
              <span className="lic">{a.license_label}</span>
              <a href={a.policy_url} target="_blank" rel="noreferrer noopener">
                {t('attribution.policyLink')}
              </a>
              <span className={`fresh${isStale ? ' stale' : ''}`}>
                {isStale
                  ? t('attribution.stale', { date: when })
                  : t('attribution.refreshed', { date: when })}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
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
