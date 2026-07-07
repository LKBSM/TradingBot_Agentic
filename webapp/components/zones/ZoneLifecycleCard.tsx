'use client';

import * as React from 'react';
import Link from 'next/link';
import { ArrowRight, ChevronDown, Eye, EyeOff } from 'lucide-react';
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
import type { Candle, FVGStatus, OBStatus } from '@/types/market-reading';
import {
  barsSince,
  buildTimeline,
  fillFraction,
  findOverlaps,
  formatDurationShort,
  priceRelation,
  type SiblingZone,
  type ZoneLifecycle,
} from '@/lib/zones/lifecycle';
import { ZoneTimeline } from './ZoneTimeline';

/**
 * One zone card — "compact d'abord, riche en dépliant". Every line is a
 * present/past FACT read off the engine payload:
 *
 * COMPACT (scannable): type/direction/status header · band (+ OB importance,
 * an engine bucket already shown on the chart) · relation to the current price
 * (inside / gap to the nearest edge) · age since formation (bars counted on the
 * real candle window, exact duration otherwise) with the boolean "testée" /
 * "pénétrée" fact · the FVG fill bar (real engine fill_level only).
 *
 * EXPANDED (chevron): the lifecycle timeline (only engine-recorded events;
 * `mitigated_at` is the FIRST contact — labelled as such, never a "×N": the
 * engine tracks no per-test history) · geometric overlaps with zones of the
 * other timeframes (pure interval intersection, phrased as geometry — never
 * "confluence"/"renforcée", mission §0).
 *
 * Actions: "Analyser la zone" (deep-link carrying the REAL engine id — the /app
 * chart re-validates it through the zone-id lock) and "Masquer / Afficher".
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

/**
 * Is the zone still worth watching? INDEPENDENT of whether it was tested. This
 * clears up the « mitigé » confusion: a mitigated OB was tapped by price but
 * still holds (still effective) — only an INVALIDATED OB (broken through) or a
 * fully FILLED FVG (imbalance closed) is spent. active / mitigated (OB) and
 * active / partially_filled (FVG) all stay effective.
 */
function effectiveness(zone: ZoneLifecycle): { effective: boolean; label: string } {
  const spent =
    zone.kind === 'ob' ? zone.status === 'invalidated' : zone.status === 'filled';
  return spent
    ? { effective: false, label: 'plus efficace' }
    : { effective: true, label: 'encore efficace' };
}

/** "prix actuellement dans la zone" / "à 3,40 pts au-dessus du prix" — fact only. */
function relationLabel(
  rel: ReturnType<typeof priceRelation>,
  instrument: string,
): string | null {
  if (!rel) return null;
  if (rel.position === 'inside') return 'prix actuellement dans la zone';
  const dist = formatPrice(rel.distance, instrument);
  return rel.position === 'above'
    ? `à ${dist} pts au-dessus du prix`
    : `à ${dist} pts en dessous du prix`;
}

/**
 * "formée il y a 26 bougies (6 h 30)" — bars from the REAL candle window when it
 * reaches back to the formation, exact duration alone otherwise. Subject is
 * "la zone" (feminine) for both kinds. Never an estimated bar count.
 */
function ageLabel(zone: ZoneLifecycle, candles: Candle[] | null, now: Date): string | null {
  const createdMs = Date.parse(zone.createdAt);
  if (Number.isNaN(createdMs)) return null;
  const elapsed = Math.max(now.getTime() - createdMs, 0);
  const duration = formatDurationShort(elapsed);
  const bars = barsSince(candles, zone.createdAt);
  if (bars == null) return `formée il y a ${duration}`;
  if (bars === 0) return 'formée sur la dernière bougie';
  return `formée il y a ${bars} bougie${bars > 1 ? 's' : ''} (${duration})`;
}

/** Boolean interaction fact — engine tracks no count, so never a "×N". */
function testedLabel(zone: ZoneLifecycle): string | null {
  if (!zone.tested) return null;
  return zone.kind === 'ob' ? 'testée' : 'pénétrée';
}

export interface ZoneLifecycleCardProps {
  zone: ZoneLifecycle;
  instrument: string;
  /**
   * Reference price for the relation badge — the unified latest price when the
   * feed serves one, the reading's close_price otherwise, null when neither is
   * available (the badge is then omitted).
   */
  referencePrice: number | null;
  /** Real candle window of the SAME combo, for the bar-count age (null = none). */
  candles: Candle[] | null;
  /** Zones of the other timeframes (same instrument) for the overlap facts. */
  siblingZones: readonly SiblingZone[];
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
  referencePrice,
  candles,
  siblingZones,
  isHidden,
  onToggleHide,
  appHref,
}: ZoneLifecycleCardProps) {
  const [expanded, setExpanded] = React.useState(false);
  const detailsId = React.useId();

  const events = buildTimeline(zone);
  const frac = zone.status === 'partially_filled' ? fillFraction(zone) : null;
  const kindLabel = zone.kind === 'ob' ? 'Order Block' : 'Fair Value Gap';

  const rel = priceRelation(zone, referencePrice);
  const relation = relationLabel(rel, instrument);
  const age = ageLabel(zone, candles, new Date());
  const tested = testedLabel(zone);
  const overlaps = findOverlaps(zone, siblingZones);

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
        <div className="flex items-center gap-1.5">
          {/* Effectiveness — the plain-language answer to "is this zone still
              usable?", so « mitigé » is never mistaken for « dead ». */}
          {(() => {
            const eff = effectiveness(zone);
            return (
              <span
                className={cn(
                  'rounded-full px-2 py-0.5 text-xs font-semibold',
                  eff.effective
                    ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400'
                    : 'bg-muted text-muted-foreground line-through decoration-1',
                )}
                title={
                  eff.effective
                    ? 'Zone toujours valable (même mitigée/testée)'
                    : 'Zone consommée (OB invalidé / FVG entièrement comblé)'
                }
              >
                {eff.label}
              </span>
            );
          })()}
          <span className="rounded-full border border-border/70 px-2 py-0.5 text-xs font-medium text-muted-foreground">
            {statusLabel(zone)}
          </span>
        </div>
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

      {/* Relation to the current price — accent 1 (amber only when inside). */}
      {relation && (
        <div>
          <span
            className={cn(
              'inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium',
              rel?.position === 'inside'
                ? 'bg-amber-500/15 text-amber-700 dark:text-amber-400'
                : 'bg-muted text-muted-foreground',
            )}
          >
            {relation}
          </span>
        </div>
      )}

      {/* Age + boolean interaction fact (no count exists — never "×N"). */}
      {age && (
        <p className="text-xs text-muted-foreground">
          {age}
          {tested ? ` · ${tested}` : ''}
        </p>
      )}

      {/* FVG fill bar — accent 2, derived from the real fill_level price. */}
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

      {/* Expand / collapse the detailed life of the zone. */}
      <button
        type="button"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
        aria-controls={detailsId}
        className="flex items-center gap-1 self-start text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
      >
        <ChevronDown
          aria-hidden
          className={cn('h-3.5 w-3.5 transition-transform', expanded && 'rotate-180')}
        />
        {expanded ? 'Réduire' : 'Détails'}
      </button>

      {expanded && (
        <div id={detailsId} className="flex flex-col gap-3 border-t border-border/60 pt-3">
          {/* Lifecycle timeline — only engine-recorded events. */}
          <ZoneTimeline events={events} />

          {/* Geometric overlaps with the other timeframes' zones (facts only). */}
          {overlaps.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-muted-foreground">
                Recoupements avec d’autres unités de temps
              </span>
              <p className="text-[11px] leading-snug text-muted-foreground/80">
                Cette zone se superpose à des zones détectées sur d’autres unités de
                temps (mêmes niveaux de prix qui se recoupent) — un repère de
                convergence, pas un signal.
              </p>
              <ul className="flex flex-col gap-0.5">
                {overlaps.map((o) => (
                  // Sibling ids are only unique WITHIN a timeframe — prefix it.
                  <li key={`${o.timeframe}-${o.id}`} className="text-xs text-muted-foreground">
                    chevauche un {o.kind === 'ob' ? 'OB' : 'FVG'} {o.timeframe}
                    {o.direction ? ` ${formatDirection(o.direction)}` : ''} (
                    {formatBand(o.levelLow, o.levelHigh, instrument)})
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="mt-1 flex flex-wrap items-center gap-2">
        <Button asChild size="sm" variant="outline">
          <Link href={appHref} aria-label="Analyser cette zone sur le graphique">
            Analyser la zone
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
