import { cn } from '@/lib/utils';
import {
  convictionLabelLong,
  directionBadgeVariant,
} from '@/lib/insight-formatters';
import type {
  ConvictionLabel,
  Direction,
  UncertaintyContext,
} from '@/types/insight';

interface ConvictionGaugeProps {
  score: number;
  label: ConvictionLabel;
  direction: Direction;
  uncertainty: UncertaintyContext;
  className?: string;
}

/**
 * Visual conviction indicator — horizontal bar 0-100 with a tick at the score,
 * a translucent overlay representing the conformal interval [lower, upper],
 * and the qualitative label underneath. Colored by direction (bull/bear/
 * neutral). Mobile-first (works down to 320px wide).
 */
export function ConvictionGauge({
  score,
  label,
  direction,
  uncertainty,
  className,
}: ConvictionGaugeProps) {
  const variant = directionBadgeVariant(direction);
  const accent =
    variant === 'bull'
      ? 'bg-sentinel-bull'
      : variant === 'bear'
        ? 'bg-sentinel-bear'
        : 'bg-sentinel-neutral';
  const accentSoft =
    variant === 'bull'
      ? 'bg-sentinel-bull/20'
      : variant === 'bear'
        ? 'bg-sentinel-bear/20'
        : 'bg-sentinel-neutral/20';

  const lower = clamp01(uncertainty.conformal_lower);
  const upper = clamp01(uncertainty.conformal_upper);
  const pct = clamp01(score);

  return (
    <div
      className={cn('w-full', className)}
      role="meter"
      aria-valuenow={score}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={`Conviction algorithmique : ${score} sur 100, ${convictionLabelLong(label)}`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-3xl font-semibold tabular-nums sm:text-4xl">
            {Math.round(score)}
          </span>
          <span className="text-sm text-muted-foreground">/ 100</span>
        </div>
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {convictionLabelLong(label)}
        </span>
      </div>

      <div className="relative mt-3 h-2 w-full overflow-hidden rounded-full bg-muted">
        {/* Conformal interval band */}
        <div
          className={cn('absolute inset-y-0 rounded-full', accentSoft)}
          style={{
            left: `${lower}%`,
            width: `${Math.max(2, upper - lower)}%`,
          }}
          aria-hidden
        />
        {/* Point estimate tick */}
        <div
          className={cn('absolute inset-y-[-2px] w-1 rounded-sm', accent)}
          style={{ left: `calc(${pct}% - 2px)` }}
          aria-hidden
        />
      </div>

      <div className="mt-2 flex items-center justify-between text-[11px] text-muted-foreground tabular-nums">
        <span>0</span>
        <span title="Intervalle conformel 90 % (Adaptive Conformal Inference)">
          marge {Math.round(lower)}–{Math.round(upper)}
        </span>
        <span>100</span>
      </div>
    </div>
  );
}

function clamp01(n: number): number {
  if (Number.isNaN(n)) return 0;
  return Math.min(100, Math.max(0, n));
}
