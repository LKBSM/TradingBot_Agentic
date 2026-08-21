'use client';

import * as React from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { useTranslations } from 'next-intl';
import { useChartView } from '@/lib/chart/viewState';
import { coerceViewActions } from '@/lib/chart/viewActions';
import { resolveComboFromQuery, buildAppHref } from '@/lib/conditions/app-link';
import {
  useCandles,
  useLatestPrice,
  useMarketReading,
} from '@/lib/market-reading/hooks';
import { useSiblingZones } from '@/lib/zones/use-sibling-zones';
import {
  DEFAULT_INSTRUMENT,
  DEFAULT_TIMEFRAME,
} from '@/lib/market-reading/perimeter';
import type { Combo } from '@/lib/market-reading/store';
import { MarketSelector } from '@/components/market/MarketSelector';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import {
  collectConsumedZones,
  collectZones,
  isConsumed,
  matchesFilter,
  sortZones,
  zonePositionGroup,
  type PositionGroup,
  type ZoneFilter,
  type ZoneLifecycle,
  type ZoneSort,
} from '@/lib/zones/lifecycle';
import { cn } from '@/lib/utils';
import { PriceFreshnessBadge } from '@/components/market-reading/PriceFreshnessBadge';
import { ZoneLifecycleCard } from './ZoneLifecycleCard';
import { ZoneMiaPanel } from './ZoneMiaPanel';

const POLL_MS = 60_000;

const FILTER_VALUES: ZoneFilter[] = ['all', 'active', 'untouched', 'consumed'];
// Factual orders only — deliberately NO importance/quality sort (mission §0).
const SORT_VALUES: ZoneSort[] = ['proximity', 'formation', 'contacts'];

// The display groups, in reading order: price inside → above → below → consumed.
type DisplayGroup = PositionGroup | 'consumed';
const GROUP_ORDER: DisplayGroup[] = ['inside', 'above', 'below', 'consumed'];

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
 * /zones — the lifecycle of every detected zone (OB / FVG) for a chosen combo,
 * grouped by position relative to the price, with a M.I.A panel whose subject is
 * the selected zone. Read-only over the SAME reading `/app` uses; it renders the
 * cycle the engine already produced and never recomputes detection.
 */
export function ZonesWorkspace({ locale }: { locale: string }) {
  const t = useTranslations('zones');
  const tApp = useTranslations('app');
  const fmt = useReadingFormatters();

  const FILTERS = FILTER_VALUES.map((value) => ({ value, label: t(`filters.${value}`) }));
  const SORTS = SORT_VALUES.map((value) => ({ value, label: t(`sorts.${value}`) }));

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

  // Single write for a market+timeframe pick from the shared MarketSelector.
  const selectCombo = React.useCallback(
    (c: Combo) => {
      setInstrumentState(c.instrument);
      setTimeframeState(c.timeframe);
      writeCombo(c.instrument, c.timeframe);
    },
    [writeCombo],
  );

  const [filter, setFilter] = React.useState<ZoneFilter>('all');
  const [sort, setSort] = React.useState<ZoneSort>('proximity');

  const { data, isLoading, isRefreshing, error, refresh } = useMarketReading(
    instrument,
    timeframe,
    { pollMs: POLL_MS },
  );

  const { change } = useLatestPrice(instrument, {
    candleCloseTs: data?.header.candle_close_ts ?? null,
  });
  const { candles } = useCandles(instrument, timeframe, {
    candleCloseTs: data?.header.candle_close_ts ?? null,
  });
  const { siblings } = useSiblingZones(instrument, timeframe);

  const { view, applyActions } = useChartView();

  const liveZones = React.useMemo(() => collectZones(data?.structure), [data]);
  const consumedZones = React.useMemo(() => collectConsumedZones(data?.structure), [data]);
  const allZones = React.useMemo(
    () => [...liveZones, ...consumedZones],
    [liveZones, consumedZones],
  );
  const liquidityPools = React.useMemo(
    () => data?.structure.liquidity_pools ?? [],
    [data],
  );

  const validZoneIds = React.useMemo(
    () => new Set(liveZones.map((z) => z.id)),
    [liveZones],
  );
  const hidden = React.useMemo(() => new Set(view.hiddenZoneIds), [view.hiddenZoneIds]);
  const referencePrice = change?.price ?? data?.header.close_price ?? null;

  // Filter, then group by position (consumed as its own group), then sort within.
  const filtered = React.useMemo(
    () => allZones.filter((z) => matchesFilter(z, filter)),
    [allZones, filter],
  );

  const groups = React.useMemo(() => {
    const out: Record<DisplayGroup, ZoneLifecycle[]> = {
      inside: [],
      above: [],
      below: [],
      consumed: [],
    };
    for (const z of filtered) {
      const g: DisplayGroup = isConsumed(z) ? 'consumed' : zonePositionGroup(z, referencePrice);
      out[g].push(z);
    }
    for (const g of GROUP_ORDER) out[g] = sortZones(out[g], sort, referencePrice);
    return out;
  }, [filtered, referencePrice, sort]);

  const renderedZones = React.useMemo(
    () => GROUP_ORDER.flatMap((g) => groups[g]),
    [groups],
  );
  const renderedZoneIds = React.useMemo(
    () => new Set(renderedZones.map((z) => z.id)),
    [renderedZones],
  );

  // ── M.I.A subject: the selected zone. Deep-link `?zone=` seeds it; otherwise
  // the first rendered zone. Stays put across polls unless it leaves the list. ──
  const zoneParam = searchParams.get('zone');
  const [selectedId, setSelectedId] = React.useState<string | null>(null);
  React.useEffect(() => {
    setSelectedId((cur) => {
      if (zoneParam && renderedZoneIds.has(zoneParam)) return zoneParam;
      if (cur && renderedZoneIds.has(cur)) return cur;
      return renderedZones[0]?.id ?? null;
    });
  }, [zoneParam, renderedZoneIds, renderedZones]);

  const selectedZone = React.useMemo(
    () => renderedZones.find((z) => z.id === selectedId) ?? null,
    [renderedZones, selectedId],
  );

  const isStaleDeepLink = Boolean(
    zoneParam && !renderedZoneIds.has(zoneParam) && !isLoading && !error,
  );
  const [dismissedStaleId, setDismissedStaleId] = React.useState<string | null>(null);
  const showStaleNotice = isStaleDeepLink && dismissedStaleId !== zoneParam;

  const cardRefs = React.useRef(new Map<string, HTMLElement | null>());
  const setCardRef = React.useCallback(
    (id: string) => (el: HTMLElement | null) => {
      if (el) cardRefs.current.set(id, el);
      else cardRefs.current.delete(id);
    },
    [],
  );

  // Scroll a deep-linked card into view once present.
  React.useEffect(() => {
    if (!zoneParam || !renderedZoneIds.has(zoneParam)) return;
    let raf = 0;
    let tries = 0;
    const tick = () => {
      const el = cardRefs.current.get(zoneParam);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
      if (tries++ < 12) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [zoneParam, renderedZoneIds]);

  const toggleHide = React.useCallback(
    (zoneId: string) => {
      const action = hidden.has(zoneId)
        ? { action: 'show_zones', params: { zone_ids: [zoneId] } }
        : { action: 'hide_zones', params: { zone_ids: [zoneId] } };
      applyActions(coerceViewActions([action], validZoneIds));
    },
    [hidden, validZoneIds, applyActions],
  );

  const showOnChart = React.useCallback(
    (zoneId: string) => {
      router.push(buildAppHref(locale, { instrument, timeframe }, zoneId));
    },
    [router, locale, instrument, timeframe],
  );

  // ── Navigate to another zone's detail from a « même endroit » item. Uses the
  // REAL engine id (never a price/label lookup) and reuses the exact deep-link
  // path (`?zone=`): the effects above then seed the M.I.A subject and scroll the
  // card into view. When the target lives on a SIBLING timeframe, we switch the
  // active timeframe too (`?timeframe=`) — the reading reloads and the same
  // deep-link resolves once its card is present. `filter='all'` reveals a target
  // the current filter would otherwise hide (else `?zone=` hits the stale notice).
  const navigateToZone = React.useCallback(
    (zoneId: string, tf: string | null) => {
      setFilter('all');
      const targetTf = tf ?? timeframe;
      if (tf && tf !== timeframe) setTimeframeState(targetTf);
      const params = new URLSearchParams(searchParams.toString());
      params.set('instrument', instrument);
      params.set('timeframe', targetTf);
      params.set('zone', zoneId);
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    },
    [searchParams, instrument, timeframe, pathname, router],
  );

  // Mobile: the M.I.A panel is a bottom sheet toggled by a button (never a panel
  // that crushes the list). Desktop: a sticky column (CSS).
  const [sheetOpen, setSheetOpen] = React.useState(false);

  const miaCtx = React.useMemo(
    () => ({
      instrument,
      price: referencePrice,
      sameTf: liveZones,
      siblings,
      pools: liquidityPools,
    }),
    [instrument, referencePrice, liveZones, siblings, liquidityPools],
  );

  const badgeSummary = `${fmt.instrument(instrument)} · ${fmt.timeframe(timeframe)} · ${t('badge.count', { count: renderedZones.length })}`;

  const cardFor = (zone: ZoneLifecycle) => (
    <ZoneLifecycleCard
      key={zone.id}
      zone={zone}
      instrument={instrument}
      referencePrice={referencePrice}
      candles={candles}
      sameTfZones={liveZones}
      siblingZones={siblings}
      liquidityPools={liquidityPools}
      isHidden={hidden.has(zone.id)}
      onToggleHide={toggleHide}
      onShowOnChart={showOnChart}
      onSelect={setSelectedId}
      onNavigateToZone={navigateToZone}
      isSelected={zone.id === selectedId}
      cardRef={setCardRef(zone.id)}
    />
  );

  return (
    <div className="pagewrap zones-wrap">
      <div className="pghead">
        <div>
          <h1>{t('title')}</h1>
          <div className="sub">{t('intro')}</div>
        </div>
        <span className="hsp" />
        <div className="flex flex-col items-end gap-1">
          <div className="livebadge">
            {isRefreshing ? (
              <span aria-live="polite">{t('refreshing')}</span>
            ) : (
              <span className="dot" aria-hidden />
            )}
            <span className="mono">{badgeSummary}</span>
          </div>
          {/* Freshness of the REFERENCE price feeding every card's proximity/gauge
              (the unified last-CLOSED price, not a live tick) — makes an apparent
              gap with the chart's live line read as elapsed time, not an error. */}
          <PriceFreshnessBadge tsSec={change?.priceTs ?? null} />
        </div>
      </div>

      {/* Controls — combo selector + filter + sort on a single wrapping row
          (UI-1b density: one row instead of two reclaims a full row of chrome
          above the list, so a second card reaches the fold). */}
      <div className="mb-1.5 flex flex-wrap items-center gap-x-4 gap-y-1.5">
        <MarketSelector
          variant="bar"
          active={{ instrument, timeframe }}
          onSelect={selectCombo}
        />
        <div className="flex items-center gap-2">
          <span className="text-[9px] font-semibold uppercase tracking-wide text-[var(--faint)]">{t('filterLabel')}</span>
          <Segmented<ZoneFilter> options={FILTERS} value={filter} onChange={setFilter} ariaLabel={t('filterAria')} />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[9px] font-semibold uppercase tracking-wide text-[var(--faint)]">{t('sortLabel')}</span>
          <Segmented<ZoneSort> options={SORTS} value={sort} onChange={setSort} ariaLabel={t('sortAria')} />
        </div>
      </div>

      {showStaleNotice && (
        <div role="status" className="zdeeplink-stale mb-4 flex items-center gap-3 rounded-md border px-3 py-2 text-[12px] text-[var(--txt)]">
          <span className="flex-1">{t('deeplink.staleNotice')}</span>
          <button type="button" className="btn" onClick={() => setDismissedStaleId(zoneParam)} aria-label={tApp('staleFocus.dismiss')}>
            {tApp('staleFocus.dismiss')}
          </button>
        </div>
      )}

      {isLoading || (isRefreshing && allZones.length === 0) ? (
        <p className="text-[12px] text-[var(--dim)]">{t('loading')}</p>
      ) : error ? (
        <div className="zone flex flex-col gap-3">
          <p className="text-[12px] text-[var(--txt)]">{t('errorMessage')}</p>
          <button type="button" className="btn self-start" onClick={refresh}>
            {t('retry')}
          </button>
        </div>
      ) : renderedZones.length === 0 ? (
        // A filter with no result → an EXPLICIT message. Never a silent fallback,
        // never a suggestion to relax the filter (mission §4).
        <p className="text-[12px] text-[var(--dim)]" data-testid="zones-empty">
          {allZones.length === 0 ? t('emptyNone') : t(`emptyFilter.${filter}`)}
        </p>
      ) : (
        <div className="zlayout">
          {/* The cards column owns the page scroll (via `.center`) — NO inner
              scroll box (the shared `.zlist` on /app keeps its own 210px box).
              Each group is its own container so the cards go 2-up when the column
              is wide enough, while the group header stays sticky within it. */}
          <div className="zcol">
            {GROUP_ORDER.filter((g) => groups[g].length > 0).map((g) => (
              <section key={g} className="zgroup" aria-label={t(`groups.${g}`)}>
                <div className="zsep">{t(`groups.${g}`)}</div>
                <div className="zcards">{groups[g].map(cardFor)}</div>
              </section>
            ))}
          </div>

          {/* Desktop: sticky panel. Mobile: bottom-sheet toggled by the button. */}
          <div className="zmia-col">
            <ZoneMiaPanel zone={selectedZone} ctx={miaCtx} />
          </div>

          <button
            type="button"
            className="zmia-fab"
            onClick={() => setSheetOpen(true)}
            aria-label={t('mia.openSheet')}
          >
            {t('mia.openSheet')}
          </button>
          {sheetOpen && (
            <div className="zmia-sheet" role="dialog" aria-label={t('mia.title')}>
              <div className="zmia-sheet-back" onClick={() => setSheetOpen(false)} />
              <div className="zmia-sheet-body">
                <button type="button" className="btn zmia-sheet-close" onClick={() => setSheetOpen(false)} aria-label={t('mia.closeSheet')}>
                  {t('mia.closeSheet')}
                </button>
                <ZoneMiaPanel zone={selectedZone} ctx={miaCtx} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
