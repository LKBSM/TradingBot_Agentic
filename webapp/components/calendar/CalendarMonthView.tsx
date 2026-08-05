'use client';

import * as React from 'react';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { ChevronRight } from 'lucide-react';
import { useLocalizedHref } from '@/lib/i18n/href';
import { useMultiFilter } from '@/lib/market-reading/use-multi-filter';
import { FilterChipGroup } from '@/components/app/FilterChipGroup';
import { useCalendarMonth } from '@/lib/calendar/useCalendar';
import { OFFICIAL_SOURCES } from '@/lib/calendar/officialSources';
import { dayKey, filterEvents, hmInZone, latestSuccess } from '@/lib/calendar/grouping';
import { parseUtc, utcOffsetLabel } from '@/lib/time/localTime';
import type {
  CalendarEvent,
  CalendarPeriodicity,
  CalendarResponse,
} from '@/types/calendar';
import '@/components/app/ui2c.css'; // reuse the shared .fchip filter chips
import './calendar.css';
import './calendar-month.css';

// Official issuing organisms only — the explicit production whitelist (CAL-1).
// ForexFactory (a private aggregator) is deliberately excluded.
const SOURCES = OFFICIAL_SOURCES;
const MARKETS = ['XAUUSD', 'EURUSD'] as const;
const PERIODICITIES: readonly CalendarPeriodicity[] = [
  'monthly', 'quarterly', 'eight_per_year',
];

/** Last IANA segment as a human label: "America/New_York" → "New York". */
function tzCity(iana: string | null): string {
  if (!iana) return '';
  const seg = iana.split('/').pop() ?? iana;
  return seg.replace(/_/g, ' ');
}

/** Short display name for a cell chip — the event's own label, unabridged. */
function shortName(ev: CalendarEvent): string {
  return ev.event;
}

/** 'YYYY-MM' for the month a Date falls in (reader-agnostic — calendar month). */
function monthKey(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
}

/** Parse 'YYYY-MM' → [year, month1..12]. */
function parseMonth(month: string): [number, number] {
  const parts = month.split('-');
  return [Number(parts[0]), Number(parts[1])];
}

/** Shift a 'YYYY-MM' by ±N months, staying valid. */
function shiftMonth(month: string, delta: number): string {
  const [y, m] = parseMonth(month);
  const d = new Date(y, (m - 1) + delta, 1);
  return monthKey(d);
}

/** All calendar-day cells for the visible month, with week starting Monday.
 *  Leading/trailing nulls pad so the 1st lands under the right weekday. */
function buildGrid(month: string): (Date | null)[] {
  const [y, m] = parseMonth(month);
  const first = new Date(y, m - 1, 1);
  const daysInMonth = new Date(y, m, 0).getDate();
  // JS getDay(): 0=Sun..6=Sat → convert to Monday-first index 0=Mon..6=Sun.
  const lead = (first.getDay() + 6) % 7;
  const cells: (Date | null)[] = [];
  for (let i = 0; i < lead; i++) cells.push(null);
  for (let day = 1; day <= daysInMonth; day++) cells.push(new Date(y, m - 1, day));
  while (cells.length % 7 !== 0) cells.push(null);
  return cells;
}

/**
 * /actualites — the scheduled-volatility calendar as a MONTHLY GRID (NW-3).
 * Announces MOMENTS, never DIRECTIONS. No amplitude sort, no impact ranking, no
 * consensus. Filters are factual: issuing organism, market, periodicity. A day
 * with no publication stays visible and empty. `data`/`now` are injectable for
 * tests; live it reads the current calendar month via useCalendarMonth.
 */
export function CalendarMonthView({
  locale,
  data: injectedData,
  now: injectedNow,
}: {
  locale: string;
  data?: CalendarResponse | null;
  now?: Date;
}) {
  const t = useTranslations('calendar');
  const now = React.useMemo(() => injectedNow ?? new Date(), [injectedNow]);

  const [month, setMonth] = React.useState<string>(() => monthKey(now));

  const hook = useCalendarMonth(month);
  const data = injectedData !== undefined ? injectedData : hook.data;
  const isLoading = injectedData !== undefined ? false : hook.isLoading;
  const error = injectedData !== undefined ? null : hook.error;

  const sourceFilter = useMultiFilter<(typeof SOURCES)[number]>(SOURCES);
  const marketFilter = useMultiFilter<(typeof MARKETS)[number]>(MARKETS);
  const periodicityFilter = useMultiFilter<CalendarPeriodicity>(PERIODICITIES);

  const [selectedDay, setSelectedDay] = React.useState<string | null>(null);

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

  // Weekday labels are a single space-joined string (next-intl message types
  // disallow arrays); split into the 7 Monday-first headers.
  const weekdays = t('month.weekdays').split(' ');

  const noneSelected =
    sourceFilter.noneSelected ||
    marketFilter.noneSelected ||
    periodicityFilter.noneSelected;

  // Events that pass the factual filters (organism/market/periodicity).
  const filtered = React.useMemo(
    () =>
      data
        ? filterEvents(
            data.events,
            sourceFilter.selected,
            marketFilter.selected,
            periodicityFilter.selected,
          )
        : [],
    [data, sourceFilter.selected, marketFilter.selected, periodicityFilter.selected],
  );

  // Index the visible-month cells and bucket events by their local day key.
  const cells = React.useMemo(() => buildGrid(month), [month]);
  const byDay = React.useMemo(() => {
    const map = new Map<string, CalendarEvent[]>();
    for (const ev of filtered) {
      const d = parseUtc(ev.scheduled_at);
      if (!d) continue;
      const key = dayKey(d);
      // Keep only events that fall inside the visible month grid.
      if (monthKey(d) !== month) continue;
      const list = map.get(key) ?? [];
      list.push(ev);
      map.set(key, list);
    }
    // Chronological within each day.
    for (const list of map.values()) {
      list.sort(
        (a, b) =>
          (parseUtc(a.scheduled_at)?.getTime() ?? 0) -
          (parseUtc(b.scheduled_at)?.getTime() ?? 0),
      );
    }
    return map;
  }, [filtered, month]);

  const dayCells = cells.filter((c): c is Date => c !== null);
  const totalThisMonth = dayCells.reduce(
    (n, d) => n + (byDay.get(dayKey(d))?.length ?? 0),
    0,
  );
  const emptyDays = dayCells.reduce(
    (n, d) => n + ((byDay.get(dayKey(d))?.length ?? 0) === 0 ? 1 : 0),
    0,
  );
  const perMarket = MARKETS.map((mk) => ({
    market: mk,
    count: dayCells.reduce(
      (n, d) =>
        n +
        (byDay.get(dayKey(d)) ?? []).filter((ev) => ev.markets.includes(mk)).length,
      0,
    ),
  }));

  const monthLabel = React.useMemo(() => {
    const [y, m] = parseMonth(month);
    return new Date(y, m - 1, 1).toLocaleDateString(locale, {
      month: 'long',
      year: 'numeric',
    });
  }, [month, locale]);

  const selectedEvents = selectedDay ? byDay.get(selectedDay) ?? [] : [];

  // Permanent proof of freshness — the most recent successful refresh across all
  // sources, shown always (not only when something is stale). NW-4 ch.5.
  const lastUpdated = React.useMemo(
    () => latestSuccess(data?.coverage.last_success),
    [data],
  );

  return (
    <div className="calm-page">
      <div className="cal-head">
        <h1>{t('title')}</h1>
        <span className="cal-badge">{t('badge')}</span>
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
      </div>

      <div className="calm-nav">
        <button
          type="button"
          className="calm-navbtn"
          onClick={() => {
            setMonth((mm) => shiftMonth(mm, -1));
            setSelectedDay(null);
          }}
        >
          {t('month.prev')}
        </button>
        <span className="calm-title mono">{monthLabel}</span>
        {lastUpdated && (
          <span className="calm-fresh">
            {t('attribution.lastUpdated', {
              date: lastUpdated.toLocaleDateString(locale),
            })}
          </span>
        )}
        <button
          type="button"
          className="calm-navbtn"
          onClick={() => {
            setMonth((mm) => shiftMonth(mm, 1));
            setSelectedDay(null);
          }}
        >
          {t('month.next')}
        </button>
        <button
          type="button"
          className="calm-navbtn calm-now"
          onClick={() => {
            setMonth(monthKey(now));
            setSelectedDay(null);
          }}
        >
          {t('month.current')}
        </button>
      </div>

      <div className="calm-body">
        <GridArea
          t={t}
          now={now}
          data={data}
          isLoading={isLoading}
          error={error}
          noneSelected={noneSelected}
          filteredCount={filtered.length}
          totalRawEvents={data?.events.length ?? 0}
          weekdays={weekdays}
          cells={cells}
          byDay={byDay}
          selectedDay={selectedDay}
          onSelectDay={setSelectedDay}
        />

        <aside className="calm-side">
          <DayPanel
            t={t}
            locale={locale}
            isLoading={isLoading}
            error={error}
            selectedDay={selectedDay}
            events={selectedEvents}
            marketName={marketName}
          />

          {/* "This month" box. A count is NEVER rendered before the data has
              loaded (CAL-1): while loading, the box shows a distinct waiting
              status, never a fabricated "0 / 31 empty days". Only once loaded
              does it assert counts — and a genuinely empty month reads as an
              explicit "no publication scheduled", not a bare zero. */}
          <div className="calm-thismonth" role="note">
            <div className="calm-tm-title">{t('month.thisMonth.title')}</div>
            {isLoading ? (
              <div className="calm-tm-status" role="status">{t('loading')}</div>
            ) : error ? (
              <div className="calm-tm-status">{t('error')}</div>
            ) : totalThisMonth > 0 ? (
              <>
                <div className="calm-tm-count mono">
                  {t('month.thisMonth.count', { count: totalThisMonth })}
                </div>
                <ul className="calm-tm-list">
                  {perMarket.map((pm) => (
                    <li key={pm.market}>
                      {t('month.thisMonth.byMarket', {
                        market: marketName(pm.market),
                        count: pm.count,
                      })}
                    </li>
                  ))}
                  <li>{t('month.thisMonth.emptyDays', { count: emptyDays })}</li>
                </ul>
                <div className="calm-tm-note">{t('month.thisMonth.note')}</div>
              </>
            ) : noneSelected ? (
              <div className="calm-tm-status">{t('empty.noSelection')}</div>
            ) : filtered.length === 0 && (data?.events.length ?? 0) > 0 ? (
              <div className="calm-tm-status">{t('month.filterEmpty')}</div>
            ) : (
              <div className="calm-tm-status">{t('month.empty')}</div>
            )}
          </div>

          {/* Honesty note — visible on the month view as on the list view: this
              calendar covers SCHEDULED moments only, never the unscheduled. */}
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
        </aside>
      </div>
    </div>
  );
}

function GridArea(props: {
  t: ReturnType<typeof useTranslations>;
  now: Date;
  data: CalendarResponse | null | undefined;
  isLoading: boolean;
  error: Error | null;
  noneSelected: boolean;
  filteredCount: number;
  totalRawEvents: number;
  weekdays: string[];
  cells: (Date | null)[];
  byDay: Map<string, CalendarEvent[]>;
  selectedDay: string | null;
  onSelectDay: (key: string) => void;
}) {
  const {
    t, now, data, isLoading, error, noneSelected, filteredCount, totalRawEvents,
    weekdays, cells, byDay, selectedDay, onSelectDay,
  } = props;

  if (isLoading) return <div className="cal-status">{t('loading')}</div>;
  if (error) return <div className="cal-status">{t('error')}</div>;
  if (!data) return <div className="cal-status">{t('error')}</div>;

  if (noneSelected) {
    return <div className="cal-empty">{t('empty.noSelection')}</div>;
  }

  const todayKey = dayKey(now);
  const monthEventCount = cells
    .filter((c): c is Date => c !== null)
    .reduce((n, d) => n + (byDay.get(dayKey(d))?.length ?? 0), 0);

  // Explicit emptiness, never silent: distinguish "no events at all this month"
  // from "filters excluded everything on an otherwise non-empty month".
  let banner: React.ReactNode = null;
  if (monthEventCount === 0) {
    banner =
      totalRawEvents === 0 ? (
        <div className="cal-empty">{t('month.empty')}</div>
      ) : filteredCount === 0 ? (
        <div className="cal-empty">{t('month.filterEmpty')}</div>
      ) : (
        // Events exist elsewhere (other months) but not on THIS month grid.
        <div className="cal-empty">{t('month.empty')}</div>
      );
  }

  return (
    <div className="calm-grid-wrap">
      <div className="calm-weekhdr">
        {weekdays.map((w) => (
          <div key={w} className="calm-wd">
            {w}
          </div>
        ))}
      </div>

      <div className="calm-grid" role="grid">
        {cells.map((cell, i) => {
          if (!cell) {
            return <div key={`blank-${i}`} className="calm-cell blank" aria-hidden />;
          }
          const key = dayKey(cell);
          const events = byDay.get(key) ?? [];
          const isToday = key === todayKey;
          const isSelected = key === selectedDay;
          const shown = events.slice(0, 2);
          const extra = events.length - shown.length;
          return (
            <button
              type="button"
              key={key}
              className={`calm-cell${isToday ? ' today' : ''}${
                isSelected ? ' selected' : ''
              }${events.length === 0 ? ' empty' : ''}`}
              aria-pressed={isSelected}
              onClick={() => onSelectDay(key)}
            >
              <span className="calm-daynum mono">{cell.getDate()}</span>
              <span className="calm-cell-events">
                {shown.map((ev) => {
                  const d = parseUtc(ev.scheduled_at);
                  const time = d ? hmInZone(d) : '';
                  return (
                    <span key={ev.event_id} className="calm-chip">
                      <span
                        className={`calm-chip-time mono${
                          ev.time_confirmed ? '' : ' unconf'
                        }`}
                        title={ev.time_confirmed ? undefined : t('timeUnconfirmed')}
                      >
                        {time}
                        {!ev.time_confirmed && (
                          <span className="calm-chip-approx" aria-hidden> ≈</span>
                        )}
                      </span>
                      <span className="calm-chip-name">{shortName(ev)}</span>
                    </span>
                  );
                })}
                {extra > 0 && (
                  <span className="calm-more">
                    {t('month.moreCount', { count: extra })}
                  </span>
                )}
              </span>
            </button>
          );
        })}
      </div>

      {banner}
    </div>
  );
}

function DayPanel({
  t,
  locale,
  isLoading,
  error,
  selectedDay,
  events,
  marketName,
}: {
  t: ReturnType<typeof useTranslations>;
  locale: string;
  isLoading: boolean;
  error: Error | null;
  selectedDay: string | null;
  events: CalendarEvent[];
  marketName: (m: string) => string;
}) {
  const lh = useLocalizedHref();
  const offsetLabel = utcOffsetLabel();

  const heading = React.useMemo(() => {
    if (!selectedDay) return '';
    const parts = selectedDay.split('-');
    const y = Number(parts[0]);
    const m = Number(parts[1]);
    const d = Number(parts[2]);
    return new Date(y, m - 1, d).toLocaleDateString(locale, {
      weekday: 'long',
      day: 'numeric',
      month: 'long',
    });
  }, [selectedDay, locale]);

  // Never assert "no publication this day" before the data has loaded (CAL-1):
  // the waiting and error states are visually distinct from an empty result.
  // (Placed after all hooks so hook order is stable across renders.)
  if (isLoading) {
    return (
      <div className="calm-panel">
        <div className="calm-panel-empty calm-panel-status" role="status">
          {t('loading')}
        </div>
      </div>
    );
  }
  if (error) {
    return (
      <div className="calm-panel">
        <div className="calm-panel-empty calm-panel-status">{t('error')}</div>
      </div>
    );
  }

  if (!selectedDay) {
    return (
      <div className="calm-panel">
        <div className="calm-panel-empty">{t('month.panel.empty')}</div>
      </div>
    );
  }

  return (
    <div className="calm-panel">
      <div className="calm-panel-head mono">{heading}</div>
      {events.length === 0 ? (
        <div className="calm-panel-empty">{t('month.panel.empty')}</div>
      ) : (
        <ul className="calm-panel-list">
          {events.map((ev) => {
            const d = parseUtc(ev.scheduled_at);
            const orgTime = d ? hmInZone(d, ev.source_timezone ?? undefined) : '';
            const localTime = d ? hmInZone(d) : '';
            const affects = ev.markets.map(marketName).join(', ');
            return (
              <li key={ev.event_id} className="calm-panel-row">
                <Link
                  className="calm-panel-link"
                  href={lh(`/actualites/${encodeURIComponent(ev.event_id)}`)}
                >
                  <div className="calm-panel-ev">{ev.event}</div>
                  <div className="calm-panel-time mono">
                    {t('month.panel.organismTime', {
                      time: orgTime,
                      city: tzCity(ev.source_timezone),
                    })}
                  </div>
                  <div className="calm-panel-time local">
                    {t('month.panel.localTime', {
                      time: localTime,
                      offset: offsetLabel,
                    })}
                  </div>
                  {!ev.time_confirmed && (
                    <div className="calm-panel-unconf">{t('timeUnconfirmed')}</div>
                  )}
                  <div className="calm-panel-meta">
                    {ev.organism ? (
                      <span>{t('provenance.organism', { organism: ev.organism })}</span>
                    ) : (
                      <span className="missing">
                        {t('provenance.organismMissing')}
                      </span>
                    )}
                    {ev.periodicity && (
                      <span>{t(`periodicity.${ev.periodicity}`)}</span>
                    )}
                    <span>{t('affects', { markets: affects })}</span>
                  </div>
                  <ChevronRight
                    className="calm-panel-chev"
                    width={13}
                    height={13}
                    aria-hidden
                  />
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
