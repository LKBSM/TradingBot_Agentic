import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import {
  formatPips,
  formatPipsRange,
  formatSignedPercent,
  formatVolatilityRegime,
} from '@/lib/insight-formatters';
import type { InsightSignalV2 } from '@/types/insight';

export function VolatilitySection({ signal }: { signal: InsightSignalV2 }) {
  const v = signal.volatility_readout;
  const expansion = v.forecast_vs_naive_pct;
  const expandTone =
    expansion >= 15
      ? 'text-sentinel-warn'
      : expansion <= -15
        ? 'text-sentinel-bull'
        : 'text-foreground';

  return (
    <AccordionItem value="volatility">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <span aria-hidden>📊</span>
          <span>Volatilité prévisionnelle</span>
        </span>
      </AccordionTrigger>
      <AccordionContent>
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary">{formatVolatilityRegime(v.regime)}</Badge>
            {v.is_fallback && (
              <Badge variant="warn" className="text-[10px]">
                Fallback ATR brut
              </Badge>
            )}
          </div>

          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Row label="Amplitude prévue" value={formatPips(v.forecast_atr_pips)} />
            <Row label="Amplitude « naïve » (ATR)" value={formatPips(v.naive_atr_pips)} />
            <Row
              label="Écart vs naïve"
              value={formatSignedPercent(expansion)}
              valueClassName={expandTone}
              className="sm:col-span-2"
              hint={
                expansion > 0
                  ? 'Le modèle anticipe une expansion par rapport à un ATR classique.'
                  : 'Le modèle anticipe une contraction par rapport à un ATR classique.'
              }
            />
            <Row
              label="Intervalle conformel"
              value={formatPipsRange(v.confidence_interval_pips)}
              className="sm:col-span-2"
              hint="Marge d'erreur calibrée par Transductive Conformal Prediction."
            />
          </dl>
        </div>
      </AccordionContent>
    </AccordionItem>
  );
}

function Row({
  label,
  value,
  hint,
  className,
  valueClassName,
}: {
  label: string;
  value: string;
  hint?: string;
  className?: string;
  valueClassName?: string;
}) {
  return (
    <div className={className}>
      <dt className="text-xs uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className={`mt-1 text-sm font-medium ${valueClassName ?? 'text-foreground'}`}>
        {value}
      </dd>
      {hint && <p className="mt-0.5 text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}
