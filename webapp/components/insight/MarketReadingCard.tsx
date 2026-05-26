import { MessageCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { ConvictionGauge } from './ConvictionGauge';
import { DisclaimerStub } from './DisclaimerStub';
import { TemporalBadge } from './TemporalBadge';
import { VerdictHeader } from './VerdictHeader';
import type { InsightSignalV2 } from '@/types/insight';

interface MarketReadingCardProps {
  signal: InsightSignalV2;
  /**
   * Optional handler for the "ask the chatbot" CTA. Wired in F4 once the
   * ChatPanel exists. In F2 the button is rendered but inert.
   */
  onAskChatbot?: () => void;
  /** Render only the hero layer (used during F2 demo). Default false. */
  heroOnly?: boolean;
}

/**
 * The central product surface — architecture progressive uniforme.
 *
 * F2 (this sprint) ships layer 1 only:
 *   - VerdictHeader (one-liner + instrument/timeframe badge)
 *   - ConvictionGauge (0-100 gauge + conformal band + label)
 *   - TemporalBadge (emitted X ago · valid for Y)
 *   - DisclaimerStub (LEGAL-PENDING placeholder)
 *   - "Demander à Sentinel" CTA (wired in F4)
 *
 * F3 will add layer 2 (5 collapsible sections: structure, regime, vol,
 * events, history) below the hero. F4 wires the chatbot. The `heroOnly`
 * prop lets us keep the card mounted as the F3 collapsibles are added
 * without breaking the F2 demo path.
 */
export function MarketReadingCard({
  signal,
  onAskChatbot,
  heroOnly: _heroOnly = false,
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
      </CardContent>
    </Card>
  );
}
