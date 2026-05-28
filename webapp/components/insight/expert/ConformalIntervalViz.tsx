import { cn } from '@/lib/utils';
import { convictionLabelLong } from '@/lib/insight-formatters';
import type {
  ConvictionLabel,
  Direction,
  UncertaintyContext,
} from '@/types/insight';

interface ConformalIntervalVizProps {
  score: number;
  label: ConvictionLabel;
  direction: Direction;
  uncertainty: UncertaintyContext;
  className?: string;
}

const BUCKET_THRESHOLDS = [0, 40, 60, 80, 100] as const;
const BUCKET_LABELS = ['weak', 'moderate', 'strong', 'institutional'] as const;

/**
 * Visualisation graphique de l'intervalle conformel — version expert.
 *
 *   ── 0 ────── 40 ────── 60 ────── 80 ────── 100
 *      │  weak  │ moderate│ strong  │ inst.   │
 *               ▼─────────████─────────▼
 *                       (lower)   (upper)
 *                          ●  <- point estimate (score)
 *
 * Utilise SVG plutôt qu'une barre HTML pour pouvoir afficher les tirets
 * de chaque seuil bucket + une bande conformelle correctement
 * positionnée. Échelle 0..100 strictement linéaire.
 */
export function ConformalIntervalViz({
  score,
  label,
  direction,
  uncertainty,
  className,
}: ConformalIntervalVizProps) {
  const tone =
    direction === 'BULLISH_SETUP'
      ? 'bull'
      : direction === 'BEARISH_SETUP'
        ? 'bear'
        : 'neutral';

  const bandColor =
    tone === 'bull'
      ? 'hsl(var(--sentinel-bull))'
      : tone === 'bear'
        ? 'hsl(var(--sentinel-bear))'
        : 'hsl(var(--sentinel-neutral))';

  const w = 480;
  const h = 96;
  const padX = 20;
  const trackY = 56;
  const trackH = 12;
  const innerW = w - 2 * padX;
  const xForScore = (s: number) => padX + (clamp01(s) / 100) * innerW;

  const lower = clamp01(uncertainty.conformal_lower);
  const upper = clamp01(uncertainty.conformal_upper);
  const point = clamp01(score);

  const bandX = xForScore(lower);
  const bandWidth = Math.max(2, xForScore(upper) - bandX);
  const pointX = xForScore(point);

  return (
    <figure
      className={cn('w-full', className)}
      aria-label={`Intervalle conformel ${uncertainty.conformal_lower.toFixed(0)} à ${uncertainty.conformal_upper.toFixed(0)}, point estimé ${score} (${convictionLabelLong(label)}).`}
    >
      <svg
        viewBox={`0 0 ${w} ${h}`}
        width="100%"
        height="auto"
        role="img"
        className="block"
      >
        {/* Bucket separators */}
        {BUCKET_THRESHOLDS.map((t) => (
          <line
            key={t}
            x1={xForScore(t)}
            x2={xForScore(t)}
            y1={trackY - 8}
            y2={trackY + trackH + 8}
            stroke="hsl(var(--border))"
            strokeWidth={1}
          />
        ))}

        {/* Bucket labels (top) */}
        {BUCKET_LABELS.map((bl, i) => {
          const startThreshold = BUCKET_THRESHOLDS[i] ?? 0;
          const endThreshold = BUCKET_THRESHOLDS[i + 1] ?? 100;
          const cx =
            (xForScore(startThreshold) + xForScore(endThreshold)) / 2;
          return (
            <text
              key={bl}
              x={cx}
              y={trackY - 14}
              textAnchor="middle"
              fontSize={10}
              fill="hsl(var(--muted-foreground))"
              className="font-medium uppercase"
            >
              {bl}
            </text>
          );
        })}

        {/* Track 0-100 */}
        <rect
          x={padX}
          y={trackY}
          width={innerW}
          height={trackH}
          rx={trackH / 2}
          fill="hsl(var(--muted))"
        />

        {/* Conformal band */}
        <rect
          x={bandX}
          y={trackY}
          width={bandWidth}
          height={trackH}
          rx={trackH / 2}
          fill={bandColor}
          fillOpacity={0.25}
        />

        {/* Band endpoints (vertical ticks) */}
        <line
          x1={bandX}
          x2={bandX}
          y1={trackY - 4}
          y2={trackY + trackH + 4}
          stroke={bandColor}
          strokeWidth={1.5}
        />
        <line
          x1={bandX + bandWidth}
          x2={bandX + bandWidth}
          y1={trackY - 4}
          y2={trackY + trackH + 4}
          stroke={bandColor}
          strokeWidth={1.5}
        />

        {/* Point estimate */}
        <circle
          cx={pointX}
          cy={trackY + trackH / 2}
          r={6}
          fill={bandColor}
          stroke="hsl(var(--background))"
          strokeWidth={2}
        />

        {/* Tick labels (bottom) — lower / point / upper */}
        <text
          x={bandX}
          y={trackY + trackH + 22}
          textAnchor="middle"
          fontSize={10}
          fill="hsl(var(--muted-foreground))"
          className="tabular-nums"
        >
          {uncertainty.conformal_lower.toFixed(0)}
        </text>
        <text
          x={pointX}
          y={trackY + trackH + 22}
          textAnchor="middle"
          fontSize={11}
          fontWeight="700"
          fill={bandColor}
          className="tabular-nums"
        >
          {Math.round(score)}
        </text>
        <text
          x={bandX + bandWidth}
          y={trackY + trackH + 22}
          textAnchor="middle"
          fontSize={10}
          fill="hsl(var(--muted-foreground))"
          className="tabular-nums"
        >
          {uncertainty.conformal_upper.toFixed(0)}
        </text>
      </svg>

      <figcaption className="mt-2 text-[11px] italic text-muted-foreground">
        Intervalle conformel à {Math.round((1 - uncertainty.coverage_alpha) * 100)} %
        — sur des conditions techniquement similaires, la probabilité réelle se
        situe statistiquement dans cette bande. Couverture empirique observée :{' '}
        {Math.round(uncertainty.empirical_coverage * 100)} % (cible{' '}
        {Math.round((1 - uncertainty.coverage_alpha) * 100)} %).
      </figcaption>
    </figure>
  );
}

function clamp01(n: number): number {
  if (Number.isNaN(n)) return 0;
  return Math.min(100, Math.max(0, n));
}
