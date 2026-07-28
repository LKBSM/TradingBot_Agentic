'use client';

import * as React from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { useTranslations } from 'next-intl';
import { useChartView } from '@/lib/chart/viewState';
import { coerceViewActions } from '@/lib/chart/viewActions';
import { resolveComboFromQuery } from '@/lib/conditions/app-link';
import {
  useCandles,
  useLatestPrice,
  useMarketReading,
} from '@/lib/market-reading/hooks';
import { useSiblingZones } from '@/lib/zones/use-sibling-zones';
import {
  DEFAULT_INSTRUMENT,
  DEFAULT_TIMEFRAME,
  DISPLAY_TIMEFRAMES,
  SUPPORTED_INSTRUMENTS,
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
  // Reuse the existing shared "dismiss this message" label (same one the /app
  // stale-focus notice uses) so this deep-link notice needs no new i18n key —
  // `zones.deeplink.staleNotice` is the only key added by the i18n workstream.
  const tApp = useTranslations('app');
  const fmt = useReadingFormatters();

  const FILTERS = FILTER_VALUES.map((value) => ({
    value,
    label: t(`filters.${value}`),
  }));
  const SORTS = SORT_VALUES.map((value) => ({
    value,
    label: t(`sorts.${value}`),
  }));

  // The combo is URL-driven (NAV-04) so `/zones?instrument=…&timeframe=…` is
  // shareable/bookmarkable and back/forward restores it — instead of a local
  // state that always reset to XAU/M15.
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const urlCombo = React.useMemo(
    () =>
      resolveComboFromQuery(
        searchParams.get('instrument') ?? undefined,
        searchParams.get('timeframe') ?? undefined,
      ),
    [searchParams],
  );

  const [instrument, setInstrumentState] = React.useState<string>(
    urlCombo?.instrument ?? DEFAULT_INSTRUMENT,
  );
  const [timeframe, setTimeframeState] = React.useState<string>(
    urlCombo?.timeframe ?? DEFAULT_TIMEFRAME,
  );

  // URL → state: reflect a deep-link / back-forward change into the selection.
  React.useEffect(() => {
    if (!urlCombo) return;
    if (urlCombo.instrument !== instrument) setInstrumentState(urlCombo.instrument);
    if (urlCombo.timeframe !== timeframe) setTimeframeState(urlCombo.timeframe);
  }, [urlCombo, instrument, timeframe]);

  const writeCombo = React.useCallback(
    (inst: string, tf: string) => {
      const params = new URLSearchParams(searchParams.toString());
      params.set('instrument', inst);
      params.set('timeframe', tf);
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    },
    [searchParams, pathname, router],
  );

  const setInstrument = React.useCallback(
    (i: string) => {
      setInstrumentState(i);
      writeCombo(i, timeframe);
    },
    [writeCombo, timeframe],
  );
  const setTimeframe = React.useCallback(
    (tf: string) => {
      setTimeframeState(tf);
      writeCombo(instrument, tf);
    },
    [writeCombo, instrument],
  );

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

  // ── Deep-link `?zone=<id>` → scroll + highlight the referenced card ────────
  // The App's "En savoir plus" opens /zones?zone=<real engine id>. The id is
  // validated against the SAME rendered list (`zones`) — an id that isn't on
  // screen (stale / no longer detected) takes the honest-notice path instead of
  // fabricating a card. `renderedZoneIds` is the set actually shown, so a zone
  // filtered out of view is treated as absent (the notice tells the truth).
  const zoneParam = searchParams.get('zone');
  const renderedZoneIds = React.useMemo(
    () => new Set(zones.map((z) => z.id)),
    [zones],
  );
  const selectedZoneId =
    zoneParam && renderedZoneIds.has(zoneParam) ? zoneParam : null;
  // A `?zone=` was requested but no matching card is on screen → stale.
  const isStaleDeepLink = Boolean(
    zoneParam && !renderedZoneIds.has(zoneParam) && !isLoading && !error,
  );

  const cardRefs = React.useRef(new Map<string, HTMLElement | null>());
  const setCardRef = React.useCallback(
    (id: string) => (el: HTMLElement | null) => {
      if (el) cardRefs.current.set(id, el);
      else cardRefs.current.delete(id);
    },
    [],
  );

  // Dismissible: closing the notice clears it until the next distinct id.
  const [dismissedStaleId, setDismissedStaleId] = React.useState<string | null>(null);
  const showStaleNotice = isStaleDeepLink && dismissedStaleId !== zoneParam;

  // Scroll the deep-linked card into view once it (and its ref) are present.
  // Retries a few frames because the list can still be settling on first paint.
  React.useEffect(() => {
    if (!selectedZoneId) return;
    let raf = 0;
    let tries = 0;
    const tick = () => {
      const el = cardRefs.current.get(selectedZoneId);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
      if (tries++ < 12) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [selectedZoneId]);

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

  // « Or · H1 · N zones » — the reference live badge summarising the combo and
  // the (filtered) zone count actually shown.
  const badgeSummary = `${fmt.instrument(instrument)} · ${fmt.timeframe(timeframe)} · ${t('badge.count', { count: zones.length })}`;

  return (
    <div className="pagewrap">
      <div className="pghead">
        <div>
          <h1>{t('title')}</h1>
          <div className="sub">{t('intro')}</div>
        </div>
        <span className="hsp" />
        <div className="livebadge">
          {isRefreshing ? (
            <span aria-live="polite">{t('refreshing')}</span>
          ) : (
            <span className="dot" aria-hidden />
          )}
          <span className="mono">{badgeSummary}</span>
        </div>
      </div>

      {/* Combo selector */}
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <Segmented
          options={SUPPORTED_INSTRUMENTS.map((i) => ({ value: i, label: fmt.instrument(i) }))}
          value={instrument}
          onChange={setInstrument}
          ariaLabel={t('selector.instrument')}
        />
        <Segmented
          options={DISPLAY_TIMEFRAMES.map((tf) => ({ value: tf, label: fmt.timeframe(tf) }))}
          value={timeframe}
          onChange={setTimeframe}
          ariaLabel={t('selector.timeframe')}
        />
      </div>

      {/* Filters + sort */}
      <div className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-2">
        <div className="flex items-center gap-2">
          <span className="text-[9px] font-semibold uppercase tracking-wide text-[var(--faint)]">{t('filterLabel')}</span>
          <Segmented<ZoneFilter> options={FILTERS} value={filter} onChange={setFilter} ariaLabel={t('filterAria')} />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[9px] font-semibold uppercase tracking-wide text-[var(--faint)]">{t('sortLabel')}</span>
          <Segmented<ZoneSort> options={SORTS} value={sort} onChange={setSort} ariaLabel={t('sortAria')} />
        </div>
      </div>

      {/* Deep-link stale notice — an honest, dismissible message when the
          requested `?zone=` id is no longer in the current reading. NEVER a
          reconstructed card. */}
      {showStaleNotice && (
        <div
          role="status"
          className="zdeeplink-stale mb-4 flex items-center gap-3 rounded-md border px-3 py-2 text-[12px] text-[var(--txt)]"
        >
          <span className="flex-1">{t('deeplink.staleNotice')}</span>
          <button
            type="button"
            className="btn"
            onClick={() => setDismissedStaleId(zoneParam)}
            aria-label={tApp('staleFocus.dismiss')}
          >
            {tApp('staleFocus.dismiss')}
          </button>
        </div>
      )}

      {/* Body */}
      {isLoading || (isRefreshing && allZones.length === 0) ? (
        // Show the loading state (not "Aucune zone") while a refresh is in flight
        // with no zones yet, so a combo switch doesn't flash an empty list before
        // the new reading lands (UI-12).
        <p className="text-[12px] text-[var(--dim)]">{t('loading')}</p>
      ) : error ? (
        <div className="zone flex flex-col gap-3">
          <p className="text-[12px] text-[var(--txt)]">{t('errorMessage')}</p>
          <button type="button" className="btn self-start" onClick={refresh}>
            {t('retry')}
          </button>
        </div>
      ) : zones.length === 0 ? (
        <p className="text-[12px] text-[var(--dim)]">
          {allZones.length === 0 ? t('emptyNone') : t('emptyFilter')}
        </p>
      ) : (
        <div>
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
              isSelected={zone.id === selectedZoneId}
              cardRef={setCardRef(zone.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
