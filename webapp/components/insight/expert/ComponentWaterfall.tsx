'use client';

import * as React from 'react';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import type { ComponentBreakdown, ComponentName } from '@/types/insight';

const COMPONENT_LABELS: Record<ComponentName, string> = {
  bos: 'Cassure de structure (BOS)',
  fvg: 'Déséquilibre (FVG)',
  ob: 'Order Block',
  regime: 'Régime de marché',
  news: 'Sentiment news',
  volume: 'Volume',
  momentum: 'Momentum (RSI + MACD)',
  rsi_divergence: 'Divergence RSI',
};

interface ComponentWaterfallProps {
  components: ReadonlyArray<ComponentBreakdown>;
  /** Aligns the colour of bars to the verdict direction (bull/bear/neutral). */
  tone?: 'bull' | 'bear' | 'neutral';
}

/**
 * Waterfall pédagogique des 8 composantes du moteur de scoring.
 *
 * Chaque barre représente une composante. La piste grise traduit son
 * poids maximal (weight_max), la portion colorée la contribution réelle.
 * Au survol/clic d'une ligne, une infobulle révèle le raisonnement
 * (champ `reasoning` du payload).
 *
 * Trié par contribution décroissante pour que les facteurs dominants
 * sautent aux yeux.
 */
export function ComponentWaterfall({
  components,
  tone = 'neutral',
}: ComponentWaterfallProps) {
  const sorted = React.useMemo(
    () => [...components].sort((a, b) => b.contribution - a.contribution),
    [components],
  );

  const colorFill =
    tone === 'bull'
      ? 'bg-sentinel-bull'
      : tone === 'bear'
        ? 'bg-sentinel-bear'
        : 'bg-sentinel-neutral';

  const totalContribution = components.reduce((s, c) => s + c.contribution, 0);
  const totalWeightMax = components.reduce((s, c) => s + c.weight_max, 0);

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Décomposition du score
        </p>
        <p className="text-xs tabular-nums text-muted-foreground">
          {totalContribution.toFixed(1)} / {totalWeightMax.toFixed(0)}
        </p>
      </div>

      <ul className="space-y-2">
        {sorted.map((c) => {
          const ratio = c.weight_max > 0 ? c.contribution / c.weight_max : 0;
          const widthPct = Math.max(0, Math.min(100, ratio * 100));
          return (
            <li key={c.name}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    className="group w-full rounded-md p-1 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    aria-label={`${COMPONENT_LABELS[c.name]} : ${c.contribution.toFixed(2)} sur ${c.weight_max.toFixed(0)}`}
                  >
                    <div className="mb-1 flex items-baseline justify-between gap-3 text-xs">
                      <span className="truncate font-medium text-foreground">
                        {COMPONENT_LABELS[c.name]}
                      </span>
                      <span className="shrink-0 tabular-nums text-muted-foreground">
                        {c.contribution.toFixed(2)} /{' '}
                        {c.weight_max.toFixed(0)}
                      </span>
                    </div>
                    <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted">
                      <div
                        className={cn(
                          'absolute inset-y-0 left-0 rounded-full transition-all group-hover:brightness-110',
                          colorFill,
                          ratio === 0 && 'opacity-30',
                        )}
                        style={{ width: `${widthPct}%` }}
                      />
                    </div>
                  </button>
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-xs text-pretty">
                  <p className="text-xs leading-relaxed">{c.reasoning}</p>
                </TooltipContent>
              </Tooltip>
            </li>
          );
        })}
      </ul>

      <p className="text-[11px] italic text-muted-foreground">
        Poids maximums : BOS 15 · FVG 15 · OB 10 · Régime 25 · News 20 ·
        Volume 10 · Momentum 3 · Divergence 2. Le score final est ensuite
        calibré (LightGBM → Isotonic → Conformal) — la somme brute ci-dessus
        n&apos;est pas exactement égale à la conviction affichée.
      </p>
    </div>
  );
}
