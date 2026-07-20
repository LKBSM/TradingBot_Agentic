'use client';

import * as React from 'react';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { ArrowRight, ChevronDown, Eye, EyeOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import type { Candle, FVGStatus, OBStatus } from '@/types/market-reading';
import {
  barsSince,
  buildTimeline,
  fillFraction,
  findOverlaps,
  formatDurationShort,
  priceRelation,
  type DurationLabels,
  type SiblingZone,
  type TimelineLabels,
  type ZoneLifecycle,
} from '@/lib/zones/lifecycle';
import { ZoneTimeline } from './ZoneTimeline';

type ZonesT = ReturnType<typeof useTranslations>;
type ReadingFmt = ReturnType<typeof useReadingFormatters>;

/**
 * One zone card — "compact d'abord, riche en dépliant". Every line is a
 * present/past FACT read off the engine payload:
 *
 * COMPACT (scannable): type/direction/status header · band · relation to the
 * current price
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
  if (direction === 'bullish') return 'text-sentinel-bull';
  if (direction === 'bearish') return 'text-sentinel-bear';
  return 'text-muted-foreground';
}

function statusLabel(zone: ZoneLifecycle, fmt: ReadingFmt): string {
  return zone.kind === 'ob'
    ? fmt.obStatus(zone.status as OBStatus)
    : fmt.fvgStatus(zone.status as FVGStatus);
}

/**
 * Descriptive « still there? » fact — NOT a prediction of future effect and
 * INDEPENDENT of whether the zone was tested. It states the present structural
 * state in the reader's own vocabulary: an OB is « invalidée » only once price
 * has CLOSED through it, an FVG « comblée » only once fully filled. A mitigated
 * OB (tapped but holding) and a partially-filled FVG are therefore still
 * « non invalidée » / « non comblée » — clearing the « mitigé » confusion
 * without ever implying the zone WILL produce an effect.
 */
function effectiveness(zone: ZoneLifecycle, t: ZonesT): {
  effective: boolean;
  label: string;
  title: string;
} {
  if (zone.kind === 'ob') {
    const spent = zone.status === 'invalidated';
    return spent
      ? {
          effective: false,
          label: t('effectiveness.obSpentLabel'),
          title: t('effectiveness.obSpentTitle'),
        }
      : {
          effective: true,
          label: t('effectiveness.obHeldLabel'),
          title: t('effectiveness.obHeldTitle'),
        };
  }
  const spent = zone.status === 'filled';
  return spent
    ? {
        effective: false,
        label: t('effectiveness.fvgSpentLabel'),
        title: t('effectiveness.fvgSpentTitle'),
      }
    : {
        effective: true,
        label: t('effectiveness.fvgHeldLabel'),
        title: t('effectiveness.fvgHeldTitle'),
      };
}

/** "prix actuellement dans la zone" / "à 3,40 pts au-dessus du prix" — fact only. */
function relationLabel(
  rel: ReturnType<typeof priceRelation>,
  instrument: string,
  t: ZonesT,
  fmt: ReadingFmt,
): string | null {
  if (!rel) return null;
  if (rel.position === 'inside') return t('relation.inside');
  const dist = fmt.price(rel.distance, instrument);
  return rel.position === 'above'
    ? t('relation.above', { dist })
    : t('relation.below', { dist });
}

/**
 * "formée il y a 26 bougies (6 h 30)" — bars from the REAL candle window when it
 * reaches back to the formation, exact duration alone otherwise. Subject is
 * "la zone" (feminine) for both kinds. Never an estimated bar count.
 */
function ageLabel(
  zone: ZoneLifecycle,
  candles: Candle[] | null,
  now: Date,
  t: ZonesT,
  durationLabels: DurationLabels,
): string | null {
  const createdMs = Date.parse(zone.createdAt);
  if (Number.isNaN(createdMs)) return null;
  const elapsed = Math.max(now.getTime() - createdMs, 0);
  const duration = formatDurationShort(elapsed, durationLabels);
  const bars = barsSince(candles, zone.createdAt);
  if (bars == null) return t('age.durationOnly', { duration });
  if (bars === 0) return t('age.lastCandle');
  return t('age.withBars', { bars, duration });
}

/** Boolean interaction fact — engine tracks no count, so never a "×N". */
function testedLabel(zone: ZoneLifecycle, t: ZonesT): string | null {
  if (!zone.tested) return null;
  return zone.kind === 'ob' ? t('tested.ob') : t('tested.fvg');
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
  const t = useTranslations('zones');
  const fmt = useReadingFormatters();
  const [expanded, setExpanded] = React.useState(false);
  const detailsId = React.useId();

  const timelineLabels: TimelineLabels = {
    formed: t('timeline.formed'),
    mitigated: t('timeline.mitigated'),
    obTested: t('timeline.obTested'),
    fvgTested: t('timeline.fvgTested'),
    filled: t('timeline.filled'),
    partial: t('timeline.partial'),
    active: t('timeline.active'),
  };
  const durationLabels: DurationLabels = {
    underMinute: t('duration.underMinute'),
    min: t('duration.min'),
    hour: t('duration.hour'),
    day: t('duration.day'),
  };

  const events = buildTimeline(zone, timelineLabels);
  const frac = zone.status === 'partially_filled' ? fillFraction(zone) : null;
  const kindLabel = zone.kind === 'ob' ? t('kind.ob') : t('kind.fvg');

  const rel = priceRelation(zone, referencePrice);
  const relation = relationLabel(rel, instrument, t, fmt);
  const age = ageLabel(zone, candles, new Date(), t, durationLabels);
  const tested = testedLabel(zone, t);
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
              {fmt.direction(zone.direction)}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          {/* Present-tense « still there? » fact (non invalidée / non comblée),
              so « mitigé » is never mistaken for « dead ». Descriptive, never a
              prediction of future effect. */}
          {(() => {
            const eff = effectiveness(zone, t);
            return (
              <span
                className={cn(
                  'rounded-full px-2 py-0.5 text-xs font-semibold',
                  eff.effective
                    ? 'bg-sentinel-bull/15 text-sentinel-bull'
                    : 'bg-muted text-muted-foreground line-through decoration-1',
                )}
                title={eff.title}
              >
                {eff.label}
              </span>
            );
          })()}
          <span className="rounded-full border border-border/70 px-2 py-0.5 text-xs font-medium text-muted-foreground">
            {statusLabel(zone, fmt)}
          </span>
        </div>
      </div>

      {/* Band — the zone's price range only. No quality/importance score is
          shown (the range itself lets the reader judge the zone's width). */}
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-sm">
        <span className="font-medium tabular-nums text-foreground">
          {fmt.band(zone.levelLow, zone.levelHigh, instrument)}
        </span>
      </div>

      {/* Relation to the current price — accent 1 (amber only when inside). */}
      {relation && (
        <div>
          <span
            className={cn(
              'inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium',
              rel?.position === 'inside'
                ? 'bg-sentinel-warn/15 text-sentinel-warn'
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
            aria-label={t('fill.aria')}
          >
            <div
              className="h-full rounded-full bg-sentinel-warn"
              style={{ width: `${Math.round(frac * 100)}%` }}
            />
          </div>
          <span className="text-xs text-muted-foreground">
            {t('fill.label', {
              price: fmt.price(zone.fillLevel as number, instrument),
              pct: Math.round(frac * 100),
            })}
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
        {expanded ? t('details.collapse') : t('details.expand')}
      </button>

      {expanded && (
        <div id={detailsId} className="flex flex-col gap-3 border-t border-border/60 pt-3">
          {/* Lifecycle timeline — only engine-recorded events. */}
          <ZoneTimeline events={events} />

          {/* Geometric overlaps with the other timeframes' zones (facts only). */}
          {overlaps.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-muted-foreground">
                {t('overlaps.heading')}
              </span>
              <p className="text-[11px] leading-snug text-muted-foreground/80">
                {t('overlaps.description')}
              </p>
              <ul className="flex flex-col gap-0.5">
                {overlaps.map((o) => (
                  // Sibling ids are only unique WITHIN a timeframe — prefix it.
                  <li key={`${o.timeframe}-${o.id}`} className="text-xs text-muted-foreground">
                    {t('overlaps.line', {
                      kind: o.kind === 'ob' ? t('kind.obShort') : t('kind.fvgShort'),
                      tf: o.timeframe,
                      dir: o.direction ? ` ${fmt.direction(o.direction)}` : '',
                      band: fmt.band(o.levelLow, o.levelHigh, instrument),
                    })}
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
          <Link href={appHref} aria-label={t('actions.analyzeAria')}>
            {t('actions.analyze')}
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
              {t('actions.show')}
            </>
          ) : (
            <>
              <EyeOff className="mr-1 h-4 w-4" aria-hidden />
              {t('actions.hide')}
            </>
          )}
        </Button>
      </div>
    </article>
  );
}
