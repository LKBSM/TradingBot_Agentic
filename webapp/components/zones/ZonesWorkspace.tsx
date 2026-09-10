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
import { useChat } from '@/components/chat/ChatProvider';
import { useLocalizedHref } from '@/lib/i18n/href';
import { ZoneLifecycleCard } from './ZoneLifecycleCard';

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
  // MIA-3 — the SINGLE M.I.A conversation (shared ChatProvider). /zones only
  // ORIENTS it: it binds the page combo (so get_market_reading targets it) and
  // sets/clears the selected-zone focus. It never owns a separate chat engine.
  const { openForCombo, setFocus } = useChat();
  const lh = useLocalizedHref();

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
  // CLN-1 §3 — once the user has deliberately deselected (clicked the selected
  // card again → null), we must NOT silently re-pick the first zone on the next
  // poll. This flag distinguishes « deliberate no-selection » from « initial
  // load, nothing chosen yet » (where auto-picking the first zone is the wanted
  // default). A deep-link (`?zone=`) always wins regardless.
  const [userTouchedSelection, setUserTouchedSelection] = React.useState(false);
  React.useEffect(() => {
    setSelectedId((cur) => {
      if (zoneParam && renderedZoneIds.has(zoneParam)) return zoneParam;
      if (cur && renderedZoneIds.has(cur)) return cur;
      if (userTouchedSelection) return null; // respect a deliberate deselection
      return renderedZones[0]?.id ?? null;
    });
  }, [zoneParam, renderedZoneIds, renderedZones, userTouchedSelection]);

  // Toggle selection: clicking the already-selected card clears the subject
  // (CLN-1 §3). Selecting is not resetting — the M.I.A conversation is kept
  // (the shared conversation is untouched), only the subject block goes away.
  const selectZone = React.useCallback((zoneId: string) => {
    setUserTouchedSelection(true);
    setSelectedId((cur) => (cur === zoneId ? null : zoneId));
  }, []);

  const selectedZone = React.useMemo(
    () => renderedZones.find((z) => z.id === selectedId) ?? null,
    [renderedZones, selectedId],
  );

  // Bind the page combo so the input is usable and M.I.A reads the right market
  // even with no zone selected (an off-zone question is answered).
  React.useEffect(() => {
    openForCombo({ instrument, timeframe });
  }, [openForCombo, instrument, timeframe]);

  // Orient by the selected zone (a REAL, clicked zone id) — or clear it. Clearing
  // the focus removes the subject block WITHOUT touching the conversation
  // (deselect ≠ reset). A question outside the zone is still answered. The label
  // is display-only (the preamble/lock uses the id); it reuses the SAME band/tag
  // as the card, no recompute.
  //
  // Idempotent by a key ref: `useReadingFormatters()` returns a fresh object each
  // render, so we must NOT let it (or any per-render value) drive setFocus — that
  // would set a new focus object every render and spin. We compute the target and
  // only push it when the (id + label) actually changes.
  const lastFocusKeyRef = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (!selectedZone) {
      if (lastFocusKeyRef.current !== null) {
        lastFocusKeyRef.current = null;
        setFocus(null);
      }
      return;
    }
    const tag = `${selectedZone.kind === 'ob' ? 'OB' : 'FVG'}${
      selectedZone.direction === 'bullish'
        ? ' ↑'
        : selectedZone.direction === 'bearish'
          ? ' ↓'
          : ''
    }`;
    const label = `${tag} · ${fmt.band(selectedZone.levelLow, selectedZone.levelHigh, instrument)}`;
    const key = `${selectedZone.id}|${label}`;
    if (lastFocusKeyRef.current === key) return;
    lastFocusKeyRef.current = key;
    setFocus({ kind: 'zone', zoneId: selectedZone.id, label });
    // `fmt` intentionally excluded: it is recreated each render (unstable) and is
    // only read to format the label above; the key guard makes re-runs no-ops.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setFocus, selectedZone, instrument]);

  // The M.I.A panel is no longer rendered inside the page — it is the shared shell
  // chat column (same component, same design as /app), fed by the focus above.
  // On unmount (leaving /zones) clear the zone focus so no stale zone subject
  // lingers in the panel on the next page.
  React.useEffect(() => () => setFocus(null), [setFocus]);

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


  const badgeSummary = `${fmt.instrument(instrument)} · ${fmt.timeframe(timeframe)} · ${t('badge.count', { count: renderedZones.length })}`;

  // VZ-4 — the zone's own page. The REAL engine id goes in the path; the combo
  // travels as query so the sheet reads the same reading (mission §4 id lock).
  const detailHref = React.useCallback(
    (zoneId: string) =>
      lh(
        `/zones/${encodeURIComponent(zoneId)}?instrument=${encodeURIComponent(
          instrument,
        )}&timeframe=${encodeURIComponent(timeframe)}`,
      ),
    [lh, instrument, timeframe],
  );

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
      onSelect={selectZone}
      onNavigateToZone={navigateToZone}
      detailHref={detailHref(zone.id)}
      isSelected={zone.id === selectedId}
      cardRef={setCardRef(zone.id)}
    />
  );

  return (
    <div className="pagewrap zones-wrap">
      <div className="pghead">
        <div>
          <h1>{t('title')}</h1>
          {/* CLN-1 §1 — the product pitch line under the title was removed: it
              belongs on the landing page, not on a work screen a subscriber
              opens every day. The context line on the right (market · timeframe
              · zone count · price time) stays — it carries facts. */}
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

        </div>
      )}
    </div>
  );
}
