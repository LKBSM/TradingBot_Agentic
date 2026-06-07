import { TrendingUp } from 'lucide-react';
import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import {
  formatHitRate,
  formatProfitFactor,
} from '@/lib/insight-formatters';
import type { InsightSignalV2 } from '@/types/insight';

/**
 * Historical track-record section — emphasised visually because it is the
 * key differentiator vs. "indicator with no provable edge" (cf. DG-072 +
 * DG-077 honest confidence positioning). When stats are absent the section
 * is still shown with a transparent explanation.
 */
export function HistorySection({ signal }: { signal: InsightSignalV2 }) {
  const h = signal.historical_stats;

  // Post-pivot (2026-05-27) the stat *values* can be null even when the
  // historical_stats object is present — quantitative claims (PF, win-rate,
  // sample size, CI) were retired from client surfaces. Treat null-valued
  // stats exactly like "no history" so the section degrades gracefully
  // instead of crashing on `null.toFixed`.
  const hasStats =
    h != null &&
    h.similar_setups_n != null &&
    h.hit_rate_observed != null &&
    h.profit_factor != null &&
    h.profit_factor_ci95 != null;

  return (
    <AccordionItem value="history">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-sentinel-bull" aria-hidden />
          <span>Historique des setups similaires</span>
          {hasStats && h && (
            <Badge variant="bull" className="ml-1 text-[10px]">
              {h.similar_setups_n} cas
            </Badge>
          )}
        </span>
      </AccordionTrigger>
      <AccordionContent>
        {!h || !hasStats ? (
          <p className="text-sm text-muted-foreground">
            Aucun historique disponible pour cette combinaison instrument /
            timeframe — le moteur n'a pas encore accumulé assez de cas
            comparables.
          </p>
        ) : (
          <div className="space-y-4">
            <div className="rounded-lg border bg-muted/30 p-4">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Sur {h.similar_setups_n} setups historiquement similaires
              </p>
              <p className="mt-2 text-2xl font-semibold tabular-nums">
                {formatProfitFactor(h.profit_factor)}€{' '}
                <span className="text-base font-normal text-muted-foreground">
                  gagnés pour 1 € perdu
                </span>
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                Intervalle de confiance 95 % :{' '}
                <span className="font-mono tabular-nums">
                  {formatProfitFactor(h.profit_factor_ci95[0])} – {formatProfitFactor(h.profit_factor_ci95[1])}
                </span>
              </p>
            </div>

            <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <Row
                label="Taux de réussite observé"
                value={formatHitRate(h.hit_rate_observed)}
              />
              <Row
                label="Couverture conformelle"
                value={`${Math.round(h.empirical_coverage * 100)} % (cible 90 %)`}
              />
              <Row
                label="Fenêtre de backtest"
                value={h.backtest_window}
                className="sm:col-span-2"
              />
            </dl>

            <p className="text-xs italic text-muted-foreground">
              Statistiques issues d'un walk-forward — performance passée
              non garante de la performance future. Aucun edge n'est
              revendiqué.
            </p>
          </div>
        )}
      </AccordionContent>
    </AccordionItem>
  );
}

function Row({
  label,
  value,
  className,
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div className={className}>
      <dt className="text-xs uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className="mt-1 text-sm font-medium text-foreground">{value}</dd>
    </div>
  );
}
