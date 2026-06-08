import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import {
  formatMarketPhase,
  formatMtfBias,
  formatMtfKey,
  formatTrend,
  formatVolatility,
  type Tone,
} from '@/lib/market-reading/formatters';
import type {
  MarketReadingRegime,
  MTFTimeframeKey,
} from '@/types/market-reading';

const TONE_TO_VARIANT: Record<Tone, 'bull' | 'bear' | 'neutral' | 'warn'> = {
  bull: 'bull',
  bear: 'bear',
  neutral: 'neutral',
  warn: 'warn',
};

// Stable display order for the MTF confluence map.
const MTF_ORDER: MTFTimeframeKey[] = ['m15', 'h1', 'h4', 'd1', 'w1'];

/**
 * Section "Régime" — restates the descriptive phase (trend / volatility /
 * market_phase) and lays out the multi-timeframe confluence map. No HMM
 * posterior, no change-point probability, no internal gate decision (those
 * niveau-2 fields were retired from client surfaces).
 */
export function RegimeSection({ regime }: { regime: MarketReadingRegime }) {
  const trend = formatTrend(regime.trend);
  const volatility = formatVolatility(regime.volatility_observed);
  const phase = formatMarketPhase(regime.market_phase);

  const mtfEntries = MTF_ORDER.filter(
    (k) => regime.mtf_confluence[k] !== undefined,
  ).map((k) => ({ key: k, bias: regime.mtf_confluence[k]! }));

  return (
    <AccordionItem value="regime">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <span aria-hidden>🌊</span>
          <span>Régime de marché</span>
        </span>
      </AccordionTrigger>
      <AccordionContent>
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={TONE_TO_VARIANT[trend.tone]}>{trend.label}</Badge>
            <span className="inline-flex items-center gap-1">
              <Badge variant={TONE_TO_VARIANT[volatility.tone]}>
                {volatility.label}
              </Badge>
              <InfoTooltip termKey="volatility" iconOnly />
            </span>
            <span className="inline-flex items-center gap-1">
              <Badge variant={TONE_TO_VARIANT[phase.tone]}>{phase.label}</Badge>
              <InfoTooltip termKey="market_phase" iconOnly />
            </span>
          </div>

          {mtfEntries.length > 0 && (
            <div>
              <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">
                <InfoTooltip termKey="mtf">Confluence multi-timeframe</InfoTooltip>
              </p>
              <div className="flex flex-wrap gap-2">
                {mtfEntries.map(({ key, bias }) => {
                  const f = formatMtfBias(bias);
                  return (
                    <span
                      key={key}
                      className="inline-flex items-center gap-1.5 rounded-md border border-border/60 px-2 py-1 text-xs"
                    >
                      <span className="font-mono font-semibold text-muted-foreground">
                        {formatMtfKey(key)}
                      </span>
                      <Badge
                        variant={TONE_TO_VARIANT[f.tone]}
                        className="text-[10px]"
                      >
                        {f.label}
                      </Badge>
                    </span>
                  );
                })}
              </div>
            </div>
          )}

          <p className="text-xs italic text-muted-foreground">
            La phase de marché décrit l’état observé. Elle ne constitue pas une
            instruction adressée au trader.
          </p>
        </div>
      </AccordionContent>
    </AccordionItem>
  );
}
