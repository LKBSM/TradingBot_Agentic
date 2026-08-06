'use client';

import * as React from 'react';
import { Loader2 } from 'lucide-react';
import { useTranslations } from 'next-intl';
import dynamic from 'next/dynamic';
import { MarketReadingCard } from '@/components/market-reading/MarketReadingCard';
import {
  ChartUnavailable,
  EmptyReadingState,
  ReadingErrorState,
  SlowLoadHint,
} from './ReadingPlaceholders';
import { ReadingSkeleton } from './ReadingSkeleton';
import { READING_DATA_SOURCE } from '@/lib/mockReadings';
import {
  useCandles,
  useLatestPrice,
  type ReadingSource,
} from '@/lib/market-reading/hooks';
import { CandlesError } from '@/lib/market-reading/api-client';
import { useLivePrice } from '@/lib/market-reading/live-price';
import { useMarketClosed } from '@/lib/market-reading/session';
import { deriveMarketStatus, type MarketStatusView } from '@/lib/market-reading/status';
import { useChartViewOptional } from '@/lib/chart/viewState';
import type { ChartSelection, ChartViewState, ReferenceLevel } from '@/lib/chart/viewActions';
import type { Combo } from '@/lib/market-reading/store';
import type { Candle, MarketReading, MarketState } from '@/types/market-reading';

/**
 * The chart is client-only (canvas) and pulls in lightweight-charts — load it
 * lazily with SSR disabled so it never runs during server render and is split
 * out of the initial bundle.
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

interface ReadingColumnProps {
  active: Combo | null;
  reading: MarketReading | null;
  isLoading: boolean;
  isRefreshing: boolean;
  error: Error | null;
  onRetry: () => void;
  /** Candle source — defaults to the module flag; forced to 'mock' in tests. */
  dataSource?: ReadingSource;
}

/**
 * Centre column — renders the detailed reading of the active combo in the
 * "Graphique d'abord" layout (chart hero → descriptive verdict → collapsible
 * sections), plus the loading / refreshing / error / empty states. The
 * "Demander à Sentinel" CTA focuses the always-present chat sidebar.
 */
export function ReadingColumn({
  active,
  reading,
  isLoading,
  isRefreshing,
  error,
  onRetry,
  dataSource = READING_DATA_SOURCE,
}: ReadingColumnProps) {
  const t = useTranslations('app');
  // Candle feed for the chart hero. Re-pulled when the combo changes or a new
  // candle closes (candle_close_ts) — never faster, to keep the load honest.
  const { candles, error: candlesError, refresh: refreshCandles } = useCandles(
    active?.instrument ?? null,
    active?.timeframe ?? null,
    { source: dataSource, candleCloseTs: reading?.header.candle_close_ts ?? null },
  );

  // Unified last price for the header — the M15 freshest close, identical
  // whatever timeframe is shown (fixes the M15-vs-H1/H4 price divergence). Pure
  // cache read; refreshes on a light interval + on each active-TF candle close.
  const { change: live } = useLatestPrice(active?.instrument ?? null, {
    source: dataSource,
    candleCloseTs: reading?.header.candle_close_ts ?? null,
  });

  // PROTOTYPE — opt-in live tick (NEXT_PUBLIC_LIVE_TICK). Streams the last price
  // for the PROVISIONAL intra-candle view: the forming candle + header price
  // move with each tick (TradingView-style), and the zone-interaction overlay
  // updates (FVG fill / OB touch). Disabled in mock mode and when the flag is
  // off — then it stays null and the chart is exactly the candle-close view.
  // Never affects detection / BOS / CHOCH (confirmed only at candle close).
  const { price: livePrice, ts: liveTs } = useLivePrice(active?.instrument ?? null, {
    enabled: dataSource === 'live' ? undefined : false,
  });

  // Chatbot-controlled DISPLAY state (layers / filters / focus / highlight).
  // Optional: outside the /app provider it defaults to "show everything", so the
  // chart's behaviour is unchanged when no chatbot action has been applied.
  const { view: chartView, selection, referenceLevel, clearSelection } =
    useChartViewOptional();

  // Deselect the active element (drop the emphasis + restore the view) when the
  // user clicks the blue box on the chart or presses Escape — same toggle as
  // re-clicking its list entry.
  const onClearHighlight = React.useCallback(() => {
    clearSelection();
  }, [clearSelection]);

  // Header price follows the tick: override the unified (closed-candle) price
  // with the live one, keeping the SAME descriptive daily reference so the % is
  // honest. Falls back to the closed-candle `live` when no tick is available.
  const liveHeader = React.useMemo(() => {
    if (live == null || livePrice == null || !Number.isFinite(livePrice)) {
      return live;
    }
    const ref = live.referenceClose;
    return {
      ...live,
      price: livePrice,
      priceTs: liveTs ?? live.priceTs,
      changeAbs: ref != null ? livePrice - ref : live.changeAbs,
      changePct: ref != null && ref !== 0 ? (livePrice - ref) / ref : live.changePct,
    };
  }, [live, livePrice, liveTs]);

  // Descriptive session state — closed spot FX / gold outside trading hours,
  // corroborated by the freshest price age (holidays). Drives the "Marché fermé"
  // badge near the price and on the chart; suppresses the "EN DIRECT" badge so
  // the app never claims to be live when it isn't. Display-only.
  // MC-1: the SERVER decides open/closed (calendar + last-candle age, DST-safe).
  // The client heuristic is only a fallback for surfaces with no server status
  // (landing mocks). We never let the browser clock override a server verdict.
  const clientClosed = useMarketClosed(
    active?.instrument ?? null,
    liveHeader?.priceTs ?? null,
  );
  const serverStatus: MarketStatusView | null = deriveMarketStatus(reading?.market_status);
  const marketClosed = serverStatus
    ? serverStatus.isClosed || serverStatus.isLagged
    : clientClosed;

  function focusChat() {
    const input = document.querySelector<HTMLTextAreaElement>(
      'textarea[aria-label="Question libre pour M.I.A Agent"]',
    );
    input?.focus();
    input?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  let body: React.ReactNode;
  if (!active) {
    body = <EmptyReadingState />;
  } else if (error) {
    body = <ReadingErrorState error={error} onRetry={onRetry} />;
  } else if (reading) {
    body = (
      <MarketReadingCard
        reading={reading}
        onAskChatbot={focusChat}
        chartSlot={buildChartSlot(reading, candles, livePrice, liveTs, active?.timeframe ?? null, chartView, onClearHighlight, marketClosed, serverStatus?.state ?? null, referenceLevel, selection, candlesError, refreshCandles)}
        live={liveHeader}
        marketClosed={marketClosed}
        status={serverStatus}
        className="w-full border-border/60 shadow-sm"
      />
    );
  } else {
    // Reading not here yet (loading, or the effect hasn't fired): skeleton now,
    // and after a threshold the "prend plus de temps" hint so a slow load never
    // reads as a frozen screen (PERF-1) — same behaviour as DesktopReading.
    body = (
      <>
        <ReadingSkeleton />
        <SlowLoadHint />
      </>
    );
  }

  return (
    <section aria-label={t('column.sectionAria')} className="min-w-0 space-y-3">
      {isRefreshing && reading && (
        <div
          className="flex items-center gap-2 text-xs text-muted-foreground"
          role="status"
          aria-live="polite"
        >
          <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
          {t('column.refreshing')}
        </div>
      )}
      {body}
    </section>
  );
}

/**
 * The chart hero for a reading: candlesticks + SMC overlays when a candle feed
 * is available, otherwise the "Graphique indisponible" placeholder (the textual
 * reading below stays usable). Candles are supplied by useCandles (live backend
 * by default, deterministic mocks in tests) — see lib/market-reading/hooks.
 */
function buildChartSlot(
  reading: MarketReading,
  candles: Candle[] | null,
  livePrice: number | null,
  liveTs: number | null,
  timeframe: string | null,
  chartView: ChartViewState,
  onClearHighlight: () => void,
  marketClosed: boolean,
  marketStatusState: MarketState | null,
  referenceLevel: ReferenceLevel | null,
  selection: ChartSelection | null,
  candlesError: Error | null = null,
  onRetryCandles?: () => void,
): React.ReactNode {
  if (!candles || candles.length === 0) {
    return (
      <ChartUnavailable
        onRetry={onRetryCandles}
        reason={candlesError instanceof CandlesError ? candlesError.reason : undefined}
      />
    );
  }
  return (
    <ReadingChart
      candles={candles}
      structure={reading.structure}
      instrument={reading.header.instrument}
      timeframe={timeframe}
      livePrice={livePrice}
      liveTs={liveTs}
      marketClosed={marketClosed}
      marketStatusState={marketStatusState}
      layers={chartView.layers}
      filter={chartView.filter}
      focus={chartView.focus}
      highlightZoneId={chartView.highlightZoneId}
      referenceLevel={referenceLevel}
      selection={selection}
      onClearHighlight={onClearHighlight}
      hiddenZoneIds={chartView.hiddenZoneIds}
      isolatedZoneIds={chartView.isolatedZoneIds}
    />
  );
}
