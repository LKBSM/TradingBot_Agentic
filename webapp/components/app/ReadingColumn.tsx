'use client';

import { Loader2 } from 'lucide-react';
import dynamic from 'next/dynamic';
import { MarketReadingCard } from '@/components/market-reading/MarketReadingCard';
import {
  ChartUnavailable,
  EmptyReadingState,
  ReadingErrorState,
} from './ReadingPlaceholders';
import { ReadingSkeleton } from './ReadingSkeleton';
import { getMockCandles } from '@/lib/mockReadings';
import type { Combo } from '@/lib/market-reading/store';
import type { MarketReading } from '@/types/market-reading';

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
}: ReadingColumnProps) {
  function focusChat() {
    const input = document.querySelector<HTMLTextAreaElement>(
      'textarea[aria-label="Question libre pour Sentinel"]',
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
        chartSlot={buildChartSlot(reading)}
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
 * reading below stays usable). Candle data is mock today — see lib/mockReadings.
 */
function buildChartSlot(reading: MarketReading): React.ReactNode {
  const candles = getMockCandles(
    reading.header.instrument,
    reading.header.timeframe,
  );
  if (!candles || candles.length === 0) {
    return <ChartUnavailable />;
  }
  return (
    <ReadingChart
      candles={candles}
      structure={reading.structure}
      instrument={reading.header.instrument}
    />
  );
}
