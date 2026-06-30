'use client';

import * as React from 'react';
import { useChartView } from '@/lib/chart/viewState';
import { coerceViewActions } from '@/lib/chart/viewActions';
import { useMarketReading } from '@/lib/market-reading/hooks';
import {
  SUPPORTED_INSTRUMENTS,
  SUPPORTED_TIMEFRAMES,
} from '@/lib/market-reading/perimeter';
import { buildAppHref } from '@/lib/conditions/app-link';
import {
  formatInstrument,
  formatTimeframe,
} from '@/lib/market-reading/formatters';
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

const FILTERS: { value: ZoneFilter; label: string }[] = [
  { value: 'all', label: 'Toutes' },
  { value: 'active', label: 'Actives' },
  { value: 'mitigated', label: 'Mitigées' },
];

const SORTS: { value: ZoneSort; label: string }[] = [
  { value: 'importance', label: 'Importance' },
  { value: 'recency', label: 'Récence' },
  { value: 'proximity', label: 'Proximité' },
];

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
  const [instrument, setInstrument] = React.useState<string>(SUPPORTED_INSTRUMENTS[0]);
  const [timeframe, setTimeframe] = React.useState<string>(SUPPORTED_TIMEFRAMES[0]);
  const [filter, setFilter] = React.useState<ZoneFilter>('all');
  const [sort, setSort] = React.useState<ZoneSort>('importance');

  const { data, isLoading, isRefreshing, error, refresh } = useMarketReading(
    instrument,
    timeframe,
    { pollMs: POLL_MS },
  );

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

  const closePrice = data?.header.close_price ?? null;

  const zones = React.useMemo(() => {
    const filtered = allZones.filter((z) => matchesFilter(z, filter));
    return sortZones(filtered, sort, closePrice);
  }, [allZones, filter, sort, closePrice]);

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
        <h1 className="text-xl font-semibold tracking-tight text-foreground">Zones</h1>
        <p className="text-sm text-muted-foreground">
          Le cycle de vie de chaque zone détectée — formation, tests, mitigation,
          comblement — décrit au présent. Lecture descriptive : aucun objectif, aucune
          prévision.
        </p>
      </header>

      {/* Combo selector */}
      <div className="flex flex-wrap items-center gap-3">
        <Segmented
          options={SUPPORTED_INSTRUMENTS.map((i) => ({ value: i, label: formatInstrument(i) }))}
          value={instrument}
          onChange={setInstrument}
          ariaLabel="Instrument"
        />
        <Segmented
          options={SUPPORTED_TIMEFRAMES.map((t) => ({ value: t, label: formatTimeframe(t) }))}
          value={timeframe}
          onChange={setTimeframe}
          ariaLabel="Unité de temps"
        />
        {isRefreshing && (
          <span className="text-xs text-muted-foreground" aria-live="polite">
            actualisation…
          </span>
        )}
      </div>

      {/* Filters + sort */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">Filtre</span>
          <Segmented<ZoneFilter> options={FILTERS} value={filter} onChange={setFilter} ariaLabel="Filtrer les zones" />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">Tri</span>
          <Segmented<ZoneSort> options={SORTS} value={sort} onChange={setSort} ariaLabel="Trier les zones" />
        </div>
      </div>

      {/* Body */}
      {isLoading ? (
        <p className="text-sm text-muted-foreground">Chargement des zones…</p>
      ) : error ? (
        <div className="space-y-3 rounded-lg border border-destructive/40 bg-destructive/5 p-4">
          <p className="text-sm text-foreground">
            Les zones ne sont pas disponibles pour cette combinaison.
          </p>
          <Button size="sm" variant="outline" onClick={refresh}>
            Réessayer
          </Button>
        </div>
      ) : zones.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {allZones.length === 0
            ? 'Aucune zone détectée sur cette combinaison.'
            : 'Aucune zone ne correspond à ce filtre.'}
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {zones.map((zone) => (
            <ZoneLifecycleCard
              key={zone.id}
              zone={zone}
              instrument={instrument}
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
