import { useTranslations } from 'next-intl';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { type Tone } from '@/lib/market-reading/formatters';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import type { MarketReadingRegime } from '@/types/market-reading';

/**
 * Market-phase panel — the descriptive replacement for the former
 * ConvictionGauge (décision 5.B.2). It shows three plain-language, colour-coded
 * labels read directly from the regime block:
 *
 *   · trend              (haussière / baissière / neutre / range)
 *   · volatility_observed (basse / normale / élevée)
 *   · market_phase        (accumulation / distribution / tendance / range / expansion)
 *
 * There is deliberately NO 0-100 score and NO conformal band — the reading
 * describes the observed phase, it does not rate a conviction (niveau 1.5).
 */

const TONE_TO_VARIANT: Record<Tone, 'bull' | 'bear' | 'neutral' | 'warn'> = {
  bull: 'bull',
  bear: 'bear',
  neutral: 'neutral',
  warn: 'warn',
};

export function MarketPhasePanel({
  regime,
  className,
}: {
  regime: MarketReadingRegime;
  className?: string;
}) {
  const t = useTranslations('reading.card');
  const fmt = useReadingFormatters();
  const trend = fmt.trend(regime.trend);
  const volatility = fmt.volatility(regime.volatility_observed);
  const phase = fmt.marketPhase(regime.market_phase);

  return (
    <div
      className={cn('w-full', className)}
      role="group"
      aria-label={t('phaseAria')}
    >
      <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {t('phaseTitle')}
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={TONE_TO_VARIANT[trend.tone]} className="text-xs">
          {trend.label}
        </Badge>
        <Badge variant={TONE_TO_VARIANT[volatility.tone]} className="text-xs">
          {volatility.label}
        </Badge>
        <Badge variant={TONE_TO_VARIANT[phase.tone]} className="text-xs">
          {phase.label}
        </Badge>
      </div>
    </div>
  );
}
