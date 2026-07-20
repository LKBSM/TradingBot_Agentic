import { useTranslations } from 'next-intl';
import { MessageCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { DisclaimerStub, EarlyAccessBadge } from '@/components/shared/DisclaimerStub';
import { cn } from '@/lib/utils';
import { MarketPhasePanel } from './MarketPhasePanel';
import { MarketReadingHeader } from './MarketReadingHeader';
import {
  MarketReadingSections,
  type MarketReadingSectionKey,
} from './MarketReadingSections';
import type { DailyChange } from '@/lib/market-reading/price';
import type { MarketReading } from '@/types/market-reading';

interface MarketReadingCardProps {
  reading: MarketReading;
  /** Opens the chatbot pre-loaded with this reading's instrument/timeframe context. */
  onAskChatbot?: () => void;
  /** Render only the hero layer (skip the collapsible sections). Default false. */
  heroOnly?: boolean;
  /** Section keys to expand on mount (default: all collapsed). */
  defaultOpenSections?: ReadonlyArray<MarketReadingSectionKey>;
  /**
   * Optional chart (or chart-unavailable placeholder) rendered just below the
   * header — the "Graphique d'abord" layout used in /app. Omitted on the
   * landing samples, which keep the text-only hero.
   */
  chartSlot?: React.ReactNode;
  /**
   * Unified last price + descriptive daily change for the header. Omitted on
   * static surfaces (landing samples), where the header shows `close_price`.
   */
  live?: DailyChange | null;
  /**
   * Descriptive session state — true when the spot market is closed. Surfaces a
   * "Marché fermé" badge next to the header price. Omitted on static surfaces.
   */
  marketClosed?: boolean;
  className?: string;
}

/**
 * Central product surface — consumes a MarketReading directly (no mapper, no
 * synthetic conviction score).
 *
 *   Layer 1 (hero, always visible):
 *     · MarketReadingHeader  · MarketPhasePanel (descriptive)
 *     · DisclaimerStub       · "Demander à Sentinel" CTA
 *
 *   Layer 2 (collapsible, default collapsed):
 *     · Structure · Régime · Événements · Synthèse
 */
export function MarketReadingCard({
  reading,
  onAskChatbot,
  heroOnly = false,
  defaultOpenSections,
  chartSlot,
  live,
  marketClosed,
  className,
}: MarketReadingCardProps) {
  const t = useTranslations('reading.card');
  return (
    <Card className={className ?? 'w-full max-w-2xl border-border/60 shadow-sm'}>
      <CardContent className="space-y-5 p-5 sm:space-y-6 sm:p-7">
        <MarketReadingHeader
          header={reading.header}
          live={live}
          marketClosed={marketClosed}
        />

        {/* "Graphique d'abord" layout (/app): the chart is framed by a discreet
            "Accès anticipé" badge in its header and, directly beneath it, the
            PERSISTENT legal mention — placed here, glued to the chart, where the
            "this is a signal" misread risk is highest. The two are deliberately
            separate: the badge is a temporary product-status marker, the line is
            the compliance disclaimer that always stays visible and legible. */}
        {chartSlot && (
          <div className="space-y-2">
            <div className="flex justify-end">
              <EarlyAccessBadge />
            </div>
            {chartSlot}
            <DisclaimerStub variant="chart" />
          </div>
        )}

        <MarketPhasePanel regime={reading.regime} />

        <div
          className={cn(
            'flex flex-col gap-3 border-t border-border/60 pt-4 sm:flex-row sm:items-center',
            // Text-only surfaces (landing samples, no chart) keep the full hero
            // disclaimer on this row; the /app chart view already shows the legal
            // mention under the chart, so this row only carries the CTA.
            chartSlot ? 'sm:justify-end' : 'sm:justify-between',
          )}
        >
          {!chartSlot && <DisclaimerStub className="sm:max-w-md" />}
          <Button
            type="button"
            variant="default"
            size="sm"
            className="w-full shrink-0 sm:w-auto"
            onClick={onAskChatbot}
            disabled={!onAskChatbot}
            aria-label={t('askChatbotAria')}
          >
            <MessageCircle aria-hidden />
            {t('askChatbot')}
          </Button>
        </div>

        {!heroOnly && (
          <MarketReadingSections
            reading={reading}
            defaultOpen={defaultOpenSections}
          />
        )}
      </CardContent>
    </Card>
  );
}
