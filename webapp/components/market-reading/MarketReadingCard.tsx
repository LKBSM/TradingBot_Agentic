import { useTranslations } from 'next-intl';
import { MessageCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { DisclaimerStub } from '@/components/shared/DisclaimerStub';
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

        {chartSlot}

        <MarketPhasePanel regime={reading.regime} />

        <div className="flex flex-col gap-3 border-t border-border/60 pt-4 sm:flex-row sm:items-center sm:justify-between">
          <DisclaimerStub className="sm:max-w-md" />
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
