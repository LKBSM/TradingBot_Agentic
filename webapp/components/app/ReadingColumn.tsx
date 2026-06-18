'use client';

import * as React from 'react';
import { Loader2 } from 'lucide-react';
import dynamic from 'next/dynamic';
import { MarketReadingCard } from '@/components/market-reading/MarketReadingCard';
import {
  ChartUnavailable,
  EmptyReadingState,
  ReadingErrorState,
} from './ReadingPlaceholders';
import { ReadingSkeleton } from './ReadingSkeleton';
import { READING_DATA_SOURCE } from '@/lib/mockReadings';
import {
  useCandles,
  useLatestPrice,
  type ReadingSource,
} from '@/lib/market-reading/hooks';
import { useLivePrice } from '@/lib/market-reading/live-price';
import type { Combo } from '@/lib/market-reading/store';
import type { Candle, MarketReading } from '@/types/market-reading';

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
        className="flex h-[280px] w-full items-center justify-center rounded-md border border-border/60 bg-muted/30 sm:h-[340px]"
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
  // Candle feed for the chart hero. Re-pulled when the combo changes or a new
  // candle closes (candle_close_ts) — never faster, to keep the load honest.
  const { candles } = useCandles(
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
  } else if (isLoading && !reading) {
    body = <ReadingSkeleton />;
  } else if (reading) {
    body = (
      <MarketReadingCard
        reading={reading}
        onAskChatbot={focusChat}
        chartSlot={buildChartSlot(reading, candles, livePrice, liveTs, active?.timeframe ?? null)}
        live={liveHeader}
        className="w-full border-border/60 shadow-sm"
      />
    );
  } else {
    body = <ReadingSkeleton />;
  }

  return (
    <section aria-label="Lecture de marché" className="min-w-0 space-y-3">
      {isRefreshing && reading && (
        <div
          className="flex items-center gap-2 text-xs text-muted-foreground"
          role="status"
          aria-live="polite"
        >
          <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
          Actualisation…
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
): React.ReactNode {
  if (!candles || candles.length === 0) {
    return <ChartUnavailable />;
  }
  return (
    <ReadingChart
      candles={candles}
      structure={reading.structure}
      instrument={reading.header.instrument}
      timeframe={timeframe}
      livePrice={livePrice}
      liveTs={liveTs}
    />
  );
}
