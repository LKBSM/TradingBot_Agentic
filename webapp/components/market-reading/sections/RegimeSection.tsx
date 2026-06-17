'use client';

import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import { formatVolatility, type Tone } from '@/lib/market-reading/formatters';
import { useMtfTrends } from '@/lib/market-reading/hooks';
import {
  MTF_TREND_ORDER,
  describeMtfAlignment,
  mtfTrendGlyph,
} from '@/lib/market-reading/mtf-trend';
import type { MarketReadingRegime } from '@/types/market-reading';

const TONE_TO_VARIANT: Record<Tone, 'bull' | 'bear' | 'neutral' | 'warn'> = {
  bull: 'bull',
  bear: 'bear',
  neutral: 'neutral',
  warn: 'warn',
};

/**
 * Section "Régime" — multi-timeframe TREND ALIGNMENT. Replaces the former
 * trend/phase restatement (which duplicated the hero) with an at-a-glance read
 * of the M15 / H1 / H4 trends + one descriptive line characterising their
 * relation. The trend values are READ from each timeframe's existing reading
 * (regime.trend) — no detection, no recompute. Volatility (a distinct, non-
 * redundant regime fact) is kept.
 */
export function RegimeSection({
  regime,
  instrument,
}: {
  regime: MarketReadingRegime;
  instrument: string;
}) {
  return (
    <AccordionItem value="regime">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <span aria-hidden>🌊</span>
          <span>Régime de marché</span>
        </span>
      </AccordionTrigger>
      <AccordionContent>
        {/* Body lives in a child so the read-only MTF fetch fires lazily — only
            when the panel is expanded (Radix unmounts collapsed content). */}
        <RegimeBody regime={regime} instrument={instrument} />
      </AccordionContent>
    </AccordionItem>
  );
}

function RegimeBody({
  regime,
  instrument,
}: {
  regime: MarketReadingRegime;
  instrument: string;
}) {
  const { trends, isLoading } = useMtfTrends(instrument);
  const volatility = formatVolatility(regime.volatility_observed);
  const description = describeMtfAlignment(trends);
  const hasAnyTrend = MTF_TREND_ORDER.some(({ key }) => trends[key] !== null);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1">
          <Badge variant={TONE_TO_VARIANT[volatility.tone]}>
            {volatility.label}
          </Badge>
          <InfoTooltip termKey="volatility" iconOnly />
        </span>
      </div>

      <div>
        <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">
          <InfoTooltip termKey="mtf">Alignement multi-timeframe</InfoTooltip>
        </p>

        {isLoading && !hasAnyTrend ? (
          <p className="text-xs text-muted-foreground">
            Lecture des timeframes…
          </p>
        ) : hasAnyTrend ? (
          <div className="flex flex-wrap items-center gap-2">
            {MTF_TREND_ORDER.map(({ key, label }) => {
              const g = mtfTrendGlyph(trends[key]);
              return (
                <Badge
                  key={key}
                  variant={TONE_TO_VARIANT[g.tone]}
                  className="font-mono tabular-nums"
                >
                  {label} {g.arrow}
                </Badge>
              );
            })}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            Alignement multi-timeframe indisponible.
          </p>
        )}

        {description && <p className="mt-2 text-sm">{description}</p>}
      </div>

      <p className="text-xs italic text-muted-foreground">
        Cet alignement décrit l’état observé des timeframes. Il ne constitue pas
        une instruction adressée au trader.
      </p>
    </div>
  );
}
