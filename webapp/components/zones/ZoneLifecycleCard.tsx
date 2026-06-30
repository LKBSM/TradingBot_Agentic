'use client';

import Link from 'next/link';
import { ArrowRight, Eye, EyeOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import {
  formatBand,
  formatDirection,
  formatFvgStatus,
  formatObImportance,
  formatObStatus,
  formatPrice,
} from '@/lib/market-reading/formatters';
import type { FVGStatus, OBStatus } from '@/types/market-reading';
import {
  buildTimeline,
  fillFraction,
  narrateZone,
  type ZoneLifecycle,
} from '@/lib/zones/lifecycle';
import { ZoneTimeline } from './ZoneTimeline';

/**
 * One zone card: type/direction badge, band, importance (OB only), current
 * state, the lifecycle timeline, an FVG fill bar (when partially filled), one
 * factual narration sentence, and the two actions — "Analyser →" (focus the
 * zone on the chart via a deep-link carrying its REAL id) and "Masquer /
 * Afficher" (toggle this zone's id in the shared chart view state). Both target
 * the engine-emitted `zone.id`; neither touches detection.
 */

function directionTone(direction: ZoneLifecycle['direction']): string {
  if (direction === 'bullish') return 'text-emerald-600 dark:text-emerald-500';
  if (direction === 'bearish') return 'text-rose-600 dark:text-rose-500';
  return 'text-muted-foreground';
}

function statusLabel(zone: ZoneLifecycle): string {
  return zone.kind === 'ob'
    ? formatObStatus(zone.status as OBStatus)
    : formatFvgStatus(zone.status as FVGStatus);
}

export interface ZoneLifecycleCardProps {
  zone: ZoneLifecycle;
  instrument: string;
  /** Whether this zone is currently hidden from the chart (shared view state). */
  isHidden: boolean;
  /** Toggle the zone's masking by its real id (routed through the id-lock). */
  onToggleHide(zoneId: string): void;
  /** Deep-link to /app focusing this zone (built with its real id). */
  appHref: string;
}

export function ZoneLifecycleCard({
  zone,
  instrument,
  isHidden,
  onToggleHide,
  appHref,
}: ZoneLifecycleCardProps) {
  const events = buildTimeline(zone);
  const frac = zone.status === 'partially_filled' ? fillFraction(zone) : null;
  const kindLabel = zone.kind === 'ob' ? 'Order Block' : 'Fair Value Gap';

  return (
    <article
      className={cn(
        'flex flex-col gap-3 rounded-lg border border-border/70 bg-card p-4 shadow-sm transition-opacity',
        isHidden && 'opacity-60',
      )}
    >
      {/* Header: type · direction · state */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="rounded-md bg-muted px-2 py-0.5 text-xs font-semibold uppercase tracking-wide text-foreground">
            {zone.kind === 'ob' ? 'OB' : 'FVG'}
          </span>
          <span className="text-sm font-semibold text-foreground">{kindLabel}</span>
          {zone.direction && (
            <span className={cn('text-sm font-medium', directionTone(zone.direction))}>
              {formatDirection(zone.direction)}
            </span>
          )}
        </div>
        <span className="rounded-full border border-border/70 px-2 py-0.5 text-xs font-medium text-muted-foreground">
          {statusLabel(zone)}
        </span>
      </div>

      {/* Band + importance */}
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-sm">
        <span className="font-medium tabular-nums text-foreground">
          {formatBand(zone.levelLow, zone.levelHigh, instrument)}
        </span>
        {zone.importance && (
          <span className="text-xs text-muted-foreground">
            importance {formatObImportance(zone.importance)}
          </span>
        )}
      </div>

      {/* FVG fill bar — derived from the real fill_level price within the band */}
      {frac != null && (
        <div className="flex flex-col gap-1">
          <div
            className="h-2 w-full overflow-hidden rounded-full bg-muted"
            role="progressbar"
            aria-valuenow={Math.round(frac * 100)}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Comblement de la zone"
          >
            <div
              className="h-full rounded-full bg-amber-500"
              style={{ width: `${Math.round(frac * 100)}%` }}
            />
          </div>
          <span className="text-xs text-muted-foreground">
            comblé jusqu’à {formatPrice(zone.fillLevel as number, instrument)} (≈{' '}
            {Math.round(frac * 100)} %)
          </span>
        </div>
      )}

      {/* Lifecycle timeline */}
      <ZoneTimeline events={events} />

      {/* Factual narration */}
      <p className="text-sm leading-relaxed text-muted-foreground">
        {narrateZone(zone, instrument)}
      </p>

      {/* Actions */}
      <div className="mt-1 flex flex-wrap items-center gap-2">
        <Button asChild size="sm" variant="outline">
          <Link href={appHref} aria-label={`Analyser cette zone sur le graphique`}>
            Analyser
            <ArrowRight className="ml-1 h-4 w-4" aria-hidden />
          </Link>
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => onToggleHide(zone.id)}
          aria-pressed={isHidden}
        >
          {isHidden ? (
            <>
              <Eye className="mr-1 h-4 w-4" aria-hidden />
              Afficher sur le graphique
            </>
          ) : (
            <>
              <EyeOff className="mr-1 h-4 w-4" aria-hidden />
              Masquer du graphique
            </>
          )}
        </Button>
      </div>
    </article>
  );
}
