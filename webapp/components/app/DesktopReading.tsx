'use client';

import * as React from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { Loader2 } from 'lucide-react';
import { useLocale, useTranslations } from 'next-intl';
import {
  ChartUnavailable,
  EmptyReadingState,
  ReadingErrorState,
  SlowLoadHint,
} from './ReadingPlaceholders';
import { ReadingSkeleton } from './ReadingSkeleton';
import { READING_DATA_SOURCE } from '@/lib/mockReadings';
import { CandlesError } from '@/lib/market-reading/api-client';
import { useRouter } from 'next/navigation';
import {
  useCandles,
  useLatestPrice,
  type ReadingSource,
} from '@/lib/market-reading/hooks';
import { useLivePrice } from '@/lib/market-reading/live-price';
import { useMarketClosed } from '@/lib/market-reading/session';
import {
  badgeLabelKey,
  deriveMarketStatus,
  formatNyTimestamp,
  type MarketStatusView,
} from '@/lib/market-reading/status';
import { useChartViewOptional } from '@/lib/chart/viewState';
import { coerceViewActions } from '@/lib/chart/viewActions';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import { useLocalizedHref } from '@/lib/i18n/href';
import { StructureCard } from './StructureCard';
import { LiquidityCard } from './LiquidityCard';
import { RegimeCard } from './RegimeCard';
import { CalendarPreview } from '@/components/calendar/CalendarPreview';
import './ui2c.css';
import type { Combo } from '@/lib/market-reading/store';
import type {
  Candle,
  MarketReading,
  MarketReadingConditions,
  MarketReadingEvents,
  MarketReadingHeader,
  MarketReadingRegime,
  MarketReadingStructure,
} from '@/types/market-reading';

/**
 * Desktop /app reading — the terminal, flat-grid reproduction of
 * docs/design/reference-desktop.html (mission UI-2). It renders the SAME data
 * the mobile ReadingColumn shows (nothing is invented) in the reference shell
 * markup: apphead · legalbar · chart pad with layer pills · a two-column panel
 * grid. Every value comes from the typed reading via useReadingFormatters — the
 * detection engine and its API are untouched.
 *
 * Desktop-only: the mobile/tablet layout (<1280px) keeps ReadingColumn +
 * MarketReadingCard (the accordion), which is the right shape for narrow widths.
 */
const ReadingChart = dynamic(
  () => import('./ReadingChart').then((m) => ({ default: m.ReadingChart })),
  {
    ssr: false,
    loading: () => (
      <div
        className="flex h-[clamp(300px,52svh,560px)] w-full items-center justify-center rounded-md border border-border/60 bg-muted/30"
        role="status"
      >
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" aria-hidden />
      </div>
    ),
  },
);

interface DesktopReadingProps {
  active: Combo | null;
  reading: MarketReading | null;
  isLoading: boolean;
  isRefreshing: boolean;
  error: Error | null;
  onRetry: () => void;
  dataSource?: ReadingSource;
}

export function DesktopReading({
  active,
  reading,
  isLoading,
  isRefreshing,
  error,
  onRetry,
  dataSource = READING_DATA_SOURCE,
}: DesktopReadingProps) {
  const t = useTranslations('app');

  // Candle + price feeds — identical wiring to ReadingColumn (mobile), so the
  // desktop layout reads exactly the same honest, candle-close data.
  const {
    candles,
    error: candlesError,
    refresh: refreshCandles,
    loadOlder,
    isLoadingOlder,
    olderError,
    reachedStart,
  } = useCandles(active?.instrument ?? null, active?.timeframe ?? null, {
    source: dataSource,
    candleCloseTs: reading?.header.candle_close_ts ?? null,
  });
  const { change: live } = useLatestPrice(active?.instrument ?? null, {
    source: dataSource,
    candleCloseTs: reading?.header.candle_close_ts ?? null,
  });
  const { price: livePrice, ts: liveTs } = useLivePrice(active?.instrument ?? null, {
    enabled: dataSource === 'live' ? undefined : false,
  });

  const { view: chartView, applyActions, selection, referenceLevel, clearSelection } =
    useChartViewOptional();

  const onClearHighlight = React.useCallback(() => {
    clearSelection();
  }, [clearSelection]);

  const liveHeader = React.useMemo(() => {
    if (live == null || livePrice == null || !Number.isFinite(livePrice)) return live;
    const ref = live.referenceClose;
    return {
      ...live,
      price: livePrice,
      priceTs: liveTs ?? live.priceTs,
      changeAbs: ref != null ? livePrice - ref : live.changeAbs,
      changePct: ref != null && ref !== 0 ? (livePrice - ref) / ref : live.changePct,
    };
  }, [live, livePrice, liveTs]);

  // MC-1: server status is authoritative; the client heuristic is a fallback.
  const clientClosed = useMarketClosed(
    active?.instrument ?? null,
    liveHeader?.priceTs ?? null,
  );
  const serverStatus = deriveMarketStatus(reading?.market_status);
  const marketClosed = serverStatus
    ? serverStatus.isClosed || serverStatus.isLagged
    : clientClosed;

  const router = useRouter();
  const lh = useLocalizedHref();

  // Page-level help state — a single open panel across the whole dashboard
  // (mission §D/E): opening one measure's or card's "?" closes any other.
  const [openHelp, setOpenHelp] = React.useState<string | null>(null);
  const onToggleHelp = React.useCallback((key: string) => {
    setOpenHelp((cur) => (cur === key ? null : key));
  }, []);

  // The id LOCK: only zone/liquidity ids the engine actually emitted may be
  // focused/highlighted. An invented id is rejected by coerceViewActions.
  const validZoneIds = React.useMemo(() => {
    const ids = new Set<string>();
    const s = reading?.structure;
    if (s) {
      for (const ob of s.order_blocks ?? []) ids.add(ob.id);
      for (const fvg of s.fair_value_gaps ?? []) ids.add(fvg.id);
      for (const pool of s.liquidity_pools ?? []) ids.add(pool.id);
    }
    return ids;
  }, [reading]);

  const selectedId = chartView.highlightZoneId;

  // Click a zone/pocket row → toggle its chart highlight through the existing
  // id-lock channel (focus + highlight); re-selecting clears it.
  const selectZone = React.useCallback(
    (id: string) => {
      if (id === selectedId) {
        applyActions(coerceViewActions([{ action: 'clear_highlight', params: {} }], validZoneIds));
        return;
      }
      applyActions(
        coerceViewActions(
          [
            { action: 'focus_zone', params: { zone_id: id } },
            { action: 'highlight_zone', params: { zone_id: id } },
          ],
          validZoneIds,
        ),
      );
    },
    [selectedId, validZoneIds, applyActions],
  );

  // "En savoir plus" → open the Zones page on this zone's card (deep-link). An
  // unknown/stale id is handled honestly by the Zones page itself.
  const openZonePage = React.useCallback(
    (id: string) => {
      const combo = active
        ? `&instrument=${active.instrument}&timeframe=${active.timeframe}`
        : '';
      router.push(lh(`/zones?zone=${encodeURIComponent(id)}${combo}`));
    },
    [active, router, lh],
  );

  if (!active) return <EmptyReadingState />;
  if (error) return <ReadingErrorState error={error} onRetry={onRetry} />;
  if (!reading)
    return (
      <>
        <ReadingSkeleton />
        <SlowLoadHint />
      </>
    );

  const header = reading.header;
  const price = liveHeader?.price ?? header.close_price;
  const changeAbs = liveHeader?.changeAbs ?? null;

  const chartSlot =
    candles && candles.length > 0 ? (
      <ReadingChart
        candles={candles}
        structure={reading.structure}
        instrument={header.instrument}
        timeframe={active.timeframe}
        livePrice={livePrice}
        liveTs={liveTs}
        marketClosed={marketClosed}
        marketStatusState={serverStatus?.state ?? null}
        layers={chartView.layers}
        filter={chartView.filter}
        focus={chartView.focus}
        highlightZoneId={chartView.highlightZoneId}
        referenceLevel={referenceLevel}
        selection={selection}
        onClearHighlight={onClearHighlight}
        hiddenZoneIds={chartView.hiddenZoneIds}
        isolatedZoneIds={chartView.isolatedZoneIds}
        analysisWindowBars={header.analysis_window_bars ?? null}
        onLoadOlder={loadOlder}
        isLoadingOlder={isLoadingOlder}
        olderError={olderError}
        reachedStart={reachedStart}
      />
    ) : (
      <ChartUnavailable
        onRetry={refreshCandles}
        reason={candlesError instanceof CandlesError ? candlesError.reason : undefined}
      />
    );

  return (
    <div className="cv-app" aria-label={t('column.sectionAria')}>
      {isRefreshing && (
        <div
          className="flex items-center gap-2 px-[18px] pt-3 text-xs text-muted-foreground"
          role="status"
          aria-live="polite"
        >
          <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
          {t('column.refreshing')}
        </div>
      )}

      <AppHead
        instrument={header.instrument}
        price={price}
        changeAbs={changeAbs}
        marketClosed={marketClosed}
        status={serverStatus}
      />

      <LegalBar />

      <div className="chartpad">
        <LayerPills
          layers={chartView.layers}
          activeOnly={chartView.filter.activeOnly}
          onToggleLayer={(layer, on) =>
            applyActions([{ action: 'set_layer_visibility', params: { layer, visible: !on } }])
          }
          onToggleMitigated={(shown) =>
            applyActions([{ action: 'filter_zones', params: { active_only: shown } }])
          }
        />
        <div className="chartbox">{chartSlot}</div>
      </div>

      <div className="panels">
        <NarratedPanel conditions={reading.conditions} />
        <RegimeCard
          regime={reading.regime}
          structure={reading.structure}
          header={header}
          price={price}
          priceTs={liveHeader?.priceTs ?? null}
          marketStatus={reading.market_status ?? null}
          referenceLevelsPayload={reading.reference_levels ?? null}
          openHelp={openHelp}
          onToggleHelp={onToggleHelp}
        />
        <StructureCard
          structure={reading.structure}
          instrument={header.instrument}
          timeframe={header.timeframe}
          price={price}
          selectedId={selectedId}
          onSelect={selectZone}
          onOpenZone={openZonePage}
          openHelp={openHelp}
          onToggleHelp={onToggleHelp}
        />
        <LiquidityCard
          structure={reading.structure}
          instrument={header.instrument}
          price={price}
          openHelp={openHelp}
          onToggleHelp={onToggleHelp}
        />
        {/* REC point 1A: the /app "Actus" panel now shows the OFFICIAL calendar's
            next releases (CalendarPreview), whose "en savoir plus" links resolve
            to real event detail pages — instead of the news-pipeline events whose
            ids never existed in the official calendar (introuvable). */}
        <CalendarPreview />
      </div>
    </div>
  );
}

/* ── Head + legal bar ─────────────────────────────────────────────────────── */

function AppHead({
  instrument,
  price,
  changeAbs,
  marketClosed,
  status,
}: {
  instrument: string;
  price: number;
  changeAbs: number | null;
  marketClosed: boolean;
  status?: MarketStatusView | null;
}) {
  const t = useTranslations('app');
  const locale = useLocale();
  const fmt = useReadingFormatters();
  const tone = fmt.changeTone(changeAbs);
  const sign = changeAbs == null || changeAbs === 0 ? '' : changeAbs > 0 ? '+' : '−';

  // Badge label: server state (closed / daily pause / data delayed) first.
  const closedLabelKey = (status && badgeLabelKey(status.state)) || 'chart.marketClosed';
  const lastClose = status ? formatNyTimestamp(status.lastCloseTs, locale) : null;
  const reopen = status ? formatNyTimestamp(status.nextOpenTs, locale) : null;
  const subline =
    status && !status.isLive
      ? status.isLagged
        ? lastClose && t('chart.noNewCandleSince', { when: lastClose })
        : [
            lastClose && t('chart.lastCandleClosed', { when: lastClose }),
            reopen && t('chart.reopensAt', { when: reopen }),
          ]
            .filter(Boolean)
            .join(' · ')
      : null;

  return (
    <div className="apphead">
      <h1>{fmt.instrument(instrument)}</h1>
      <span className="price mono">{fmt.price(price, instrument)}</span>
      {changeAbs != null && (
        <span
          className={`mono ${tone === 'bull' ? 'up' : tone === 'bear' ? 'dn' : ''}`}
          style={{ fontSize: '12px', fontWeight: 600 }}
        >
          {sign}
          {fmt.price(Math.abs(changeAbs), instrument)}
        </span>
      )}
      <span className="hsp" />
      {marketClosed ? (
        <div className="livebadge" role="status" title={subline || undefined}>
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: '50%',
              background: 'var(--faint)',
              flexShrink: 0,
            }}
            aria-hidden
          />
          <span className="mono">{t(closedLabelKey)}</span>
        </div>
      ) : (
        <div className="livebadge" role="status">
          <span className="dot" aria-hidden />
          <span className="mono">{t('desktop.live')}</span>
        </div>
      )}
    </div>
  );
}

function LegalBar() {
  const t = useTranslations('app');
  return (
    <div className="legalbar">
      <span className="ea">
        <span className="d" aria-hidden />
        {t('desktop.earlyAccess')}
      </span>
      <span className="legal-inline">
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M12 3l8 4v5c0 4-3.4 7.4-8 9-4.6-1.6-8-5-8-9V7l8-4z" />
        </svg>
        {t('desktop.legalInline')}
      </span>
    </div>
  );
}

/* ── Layer pills ──────────────────────────────────────────────────────────── */

function LayerPills({
  layers,
  activeOnly,
  onToggleLayer,
  onToggleMitigated,
}: {
  layers: { ob: boolean; fvg: boolean; liquidity: boolean; breaks: boolean };
  activeOnly: boolean;
  onToggleLayer: (layer: 'ob' | 'fvg' | 'liquidity' | 'breaks', on: boolean) => void;
  onToggleMitigated: (shown: boolean) => void;
}) {
  const t = useTranslations('app');
  // "Mitigées" ON = mitigated/tested zones shown = NOT active-only.
  const mitigatedShown = !activeOnly;
  const single: Array<{
    key: 'ob' | 'fvg' | 'liquidity' | 'breaks';
    label: string;
    sw: string;
    on: boolean;
  }> = [
    { key: 'ob', label: t('layers.ob'), sw: 'var(--ob-l)', on: layers.ob },
    { key: 'fvg', label: t('layers.fvg'), sw: 'var(--fvg-l)', on: layers.fvg },
    { key: 'liquidity', label: t('layers.liquidity'), sw: 'var(--liq)', on: layers.liquidity },
    { key: 'breaks', label: t('layers.breaks'), sw: 'var(--dim)', on: layers.breaks },
  ];
  return (
    <div className="layerrow">
      {single.map((l) => (
        <button
          key={l.key}
          type="button"
          className={`layer${l.on ? '' : ' off'}`}
          aria-pressed={l.on}
          onClick={() => onToggleLayer(l.key, l.on)}
        >
          <span className="sw" style={{ background: l.sw }} aria-hidden />
          {l.label}
        </button>
      ))}
      <button
        type="button"
        className={`layer${mitigatedShown ? '' : ' off'}`}
        aria-pressed={mitigatedShown}
        onClick={() => onToggleMitigated(mitigatedShown)}
      >
        <span className="sw" style={{ background: 'var(--faint)' }} aria-hidden />
        {t('layers.mitigated')}
      </button>
    </div>
  );
}

/* ── Panels ───────────────────────────────────────────────────────────────── */

function NarratedPanel({ conditions }: { conditions: MarketReadingConditions }) {
  const t = useTranslations('app');
  const paragraphs = conditions.description
    .split(/\n+/)
    .map((p) => p.trim())
    .filter(Boolean);
  return (
    <div className="card wide">
      <div className="card-h">
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M4 6h16M4 12h16M4 18h10" />
        </svg>
        <h3>{t('desktop.narratedTitle')}</h3>
        <span className="badge2">{t('desktop.narratedBadge')}</span>
      </div>
      <div className="narr">
        {paragraphs.length > 0 ? (
          paragraphs.map((p, i) => <p key={i}>{p}</p>)
        ) : (
          <p>{conditions.description}</p>
        )}
      </div>
      <div className="narrfoot">
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M9 12l2 2 4-4" />
          <circle cx="12" cy="12" r="9" />
        </svg>
        {t('desktop.narratedFooter')}
      </div>
    </div>
  );
}

// NewsPanel (forexfactory news_upcoming) removed in REC point 1A: its deep-links
// pointed at ids that never existed in the official calendar (introuvable). The
// /app "Actus" panel is now <CalendarPreview />, sourced from the official
// calendar, whose "en savoir plus" links resolve to real event detail pages.
