'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import { useChartView } from '@/lib/chart/viewState';
import { coerceViewActions } from '@/lib/chart/viewActions';
import {
  useCandles,
  useLatestPrice,
  useMarketReading,
} from '@/lib/market-reading/hooks';
import { useSiblingZones } from '@/lib/zones/use-sibling-zones';
import {
  SUPPORTED_INSTRUMENTS,
  SUPPORTED_TIMEFRAMES,
} from '@/lib/market-reading/perimeter';
import { buildAppHref } from '@/lib/conditions/app-link';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import {
  collectZones,
  matchesFilter,
  sortZones,
  type ZoneFilter,
  type ZoneSort,
} from '@/lib/zones/lifecycle';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { ZoneLifecycleCard } from './ZoneLifecycleCard';

const POLL_MS = 60_000;

const FILTER_VALUES: ZoneFilter[] = ['all', 'active', 'mitigated'];

// Factual orders only — deliberately NO importance/quality sort (a "strength"
// ranking would be an implicit recommendation, mission §0).
const SORT_VALUES: ZoneSort[] = ['proximity', 'recency', 'state'];

/** Small segmented control (display-only, no detection impact). */
function Segmented<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange(v: T): void;
  ariaLabel: string;
}) {
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      className="inline-flex flex-wrap gap-1 rounded-md border border-border/70 p-1"
    >
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          aria-pressed={value === o.value}
          className={cn(
            'rounded px-2.5 py-1 text-xs font-medium transition-colors',
            value === o.value
              ? 'bg-foreground text-background'
              : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/**
 * /zones — the lifecycle of every detected zone (OB / FVG) for a chosen combo.
 * Read-only over the SAME reading the /app surface uses (`useMarketReading`); it
 * renders the cycle the engine already produced and never recomputes detection.
 * "Masquer" and "Analyser" both act through the shared chart view state so the
 * effect is reflected on the chart (`/app`).
 */
export function ZonesWorkspace({ locale }: { locale: string }) {
  const t = useTranslations('zones');
  const fmt = useReadingFormatters();

  const FILTERS = FILTER_VALUES.map((value) => ({
    value,
    label: t(`filters.${value}`),
  }));
  const SORTS = SORT_VALUES.map((value) => ({
    value,
    label: t(`sorts.${value}`),
  }));

  const [instrument, setInstrument] = React.useState<string>(SUPPORTED_INSTRUMENTS[0]);
  const [timeframe, setTimeframe] = React.useState<string>(SUPPORTED_TIMEFRAMES[0]);
  const [filter, setFilter] = React.useState<ZoneFilter>('all');
  // Proximity to the price is the default order — the most useful factual one.
  const [sort, setSort] = React.useState<ZoneSort>('proximity');

  const { data, isLoading, isRefreshing, error, refresh } = useMarketReading(
    instrument,
    timeframe,
    { pollMs: POLL_MS },
  );

  // Freshest unified price (M15 cache read, light poll) for the relation badge
  // and the proximity sort; the reading's close_price is the fallback. Both are
  // engine facts — never a projection.
  const { change } = useLatestPrice(instrument, {
    candleCloseTs: data?.header.candle_close_ts ?? null,
  });

  // Real candle window of the combo — the bar-count part of the zone age is
  // counted on it (never derived by dividing elapsed time by the timeframe).
  const { candles } = useCandles(instrument, timeframe, {
    candleCloseTs: data?.header.candle_close_ts ?? null,
  });

  // Zones of the other timeframes (same instrument) for the overlap facts.
  const { siblings } = useSiblingZones(instrument, timeframe);

  const { view, applyActions } = useChartView();

  const allZones = React.useMemo(() => collectZones(data?.structure), [data]);

  // The id lock: the ONLY zones a hide/show may reference are the ones the engine
  // emitted in THIS reading — identical to AppWorkspace's set. An invented id is
  // rejected by `coerceViewActions`, so it masks nothing.
  const validZoneIds = React.useMemo(
    () => new Set(allZones.map((z) => z.id)),
    [allZones],
  );

  const hidden = React.useMemo(
    () => new Set(view.hiddenZoneIds),
    [view.hiddenZoneIds],
  );

  // useLatestPrice first (freshest closed price), close_price as the fallback.
  const referencePrice = change?.price ?? data?.header.close_price ?? null;

  const zones = React.useMemo(() => {
    const filtered = allZones.filter((z) => matchesFilter(z, filter));
    return sortZones(filtered, sort, referencePrice);
  }, [allZones, filter, sort, referencePrice]);

  const toggleHide = React.useCallback(
    (zoneId: string) => {
      const action = hidden.has(zoneId)
        ? { action: 'show_zones', params: { zone_ids: [zoneId] } }
        : { action: 'hide_zones', params: { zone_ids: [zoneId] } };
      // Routed through the SAME Couche-4 coercion as the chat/chart: a stale or
      // invented id is dropped before it can reach the shared view state.
      const coerced = coerceViewActions([action], validZoneIds);
      applyActions(coerced);
    },
    [hidden, validZoneIds, applyActions],
  );

  return (
    <section className="flex flex-col gap-5">
      <header className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold tracking-tight text-foreground">{t('title')}</h1>
        <p className="text-sm text-muted-foreground">{t('intro')}</p>
      </header>

      {/* Combo selector */}
      <div className="flex flex-wrap items-center gap-3">
        <Segmented
          options={SUPPORTED_INSTRUMENTS.map((i) => ({ value: i, label: fmt.instrument(i) }))}
          value={instrument}
          onChange={setInstrument}
          ariaLabel={t('selector.instrument')}
        />
        <Segmented
          options={SUPPORTED_TIMEFRAMES.map((tf) => ({ value: tf, label: fmt.timeframe(tf) }))}
          value={timeframe}
          onChange={setTimeframe}
          ariaLabel={t('selector.timeframe')}
        />
        {isRefreshing && (
          <span className="text-xs text-muted-foreground" aria-live="polite">
            {t('refreshing')}
          </span>
        )}
      </div>

      {/* Filters + sort */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">{t('filterLabel')}</span>
          <Segmented<ZoneFilter> options={FILTERS} value={filter} onChange={setFilter} ariaLabel={t('filterAria')} />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">{t('sortLabel')}</span>
          <Segmented<ZoneSort> options={SORTS} value={sort} onChange={setSort} ariaLabel={t('sortAria')} />
        </div>
      </div>

      {/* Body */}
      {isLoading ? (
        <p className="text-sm text-muted-foreground">{t('loading')}</p>
      ) : error ? (
        <div className="space-y-3 rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm text-foreground">{t('errorMessage')}</p>
          <Button size="sm" variant="outline" onClick={refresh}>
            {t('retry')}
          </Button>
        </div>
      ) : zones.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {allZones.length === 0 ? t('emptyNone') : t('emptyFilter')}
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {zones.map((zone) => (
            <ZoneLifecycleCard
              key={zone.id}
              zone={zone}
              instrument={instrument}
              referencePrice={referencePrice}
              candles={candles}
              siblingZones={siblings}
              isHidden={hidden.has(zone.id)}
              onToggleHide={toggleHide}
              appHref={buildAppHref(locale, { instrument, timeframe }, zone.id)}
            />
          ))}
        </div>
      )}
    </section>
  );
}
