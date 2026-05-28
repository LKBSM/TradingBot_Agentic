import { MessageCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { ConvictionGauge } from './ConvictionGauge';
import { DisclaimerStub } from './DisclaimerStub';
import { InsightSections, type InsightSectionKey } from './InsightSections';
import { TemporalBadge } from './TemporalBadge';
import { VerdictHeader } from './VerdictHeader';
import type { InsightSignalV2 } from '@/types/insight';

interface MarketReadingCardProps {
  signal: InsightSignalV2;
  /**
   * Optional handler for the "ask the chatbot" CTA. Wired in F4 once the
   * ChatPanel exists. In F2/F3 the button is rendered but inert.
   */
  onAskChatbot?: () => void;
  /** Render only the hero layer (skip the six collapsibles). Default false. */
  heroOnly?: boolean;
  /** Section keys to expand on mount (default: all collapsed). */
  defaultOpenSections?: ReadonlyArray<InsightSectionKey>;
}

/**
 * Central product surface — architecture progressive uniforme.
 *
 *   Layer 1 (hero, always visible):
 *     · VerdictHeader  · ConvictionGauge  · TemporalBadge
 *     · DisclaimerStub · "Demander à Sentinel" CTA
 *
 *   Layer 2 (collapsible, default collapsed):
 *     · 📐 Structure · 🌊 Régime · 📊 Volatilité · 📅 Événements
 *     · 📈 Historique (visually highlighted — hero differentiator)
 *
 *   Layer 3 (chatbot, F4) — invoked from the CTA.
 */
export function MarketReadingCard({
  signal,
  onAskChatbot,
  heroOnly = false,
  defaultOpenSections,
}: MarketReadingCardProps) {
  return (
    <Card className="w-full max-w-2xl border-border/60 shadow-sm">
      <CardContent className="space-y-5 p-5 sm:space-y-6 sm:p-7">
        <VerdictHeader signal={signal} />

        <ConvictionGauge
          score={signal.conviction_0_100}
          label={signal.conviction_label}
          direction={signal.direction}
          uncertainty={signal.uncertainty}
        />

        <TemporalBadge
          createdAtUtc={signal.created_at_utc}
          validUntilUtc={signal.valid_until_utc}
        />

        <div className="flex flex-col gap-3 border-t border-border/60 pt-4 sm:flex-row sm:items-center sm:justify-between">
          <DisclaimerStub className="sm:max-w-md" />
          <Button
            type="button"
            variant="default"
            size="sm"
            className="w-full shrink-0 sm:w-auto"
            onClick={onAskChatbot}
            disabled={!onAskChatbot}
            aria-label="Ouvrir le chatbot pour poser une question contextuelle"
          >
            <MessageCircle aria-hidden />
            Demander à Sentinel
          </Button>
        </div>

        {!heroOnly && (
          <InsightSections signal={signal} defaultOpen={defaultOpenSections} />
        )}
      </CardContent>
    </Card>
  );
}
