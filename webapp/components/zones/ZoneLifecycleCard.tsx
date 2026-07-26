'use client';

import * as React from 'react';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { ChevronDown } from 'lucide-react';
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
 * One zone card — UI-2 terminal reference (`.zone`). Every line is a present/past
 * FACT read off the engine payload, restyled onto the shared design tokens:
 *
 *   · `.ztop` — a `.tagx` type/direction tag (OB↑ / OB↓ / FVG↑ / FVG↓, bull/bear
 *     by direction), the price band as a mono `.rng`, and a `.zstate` badge
 *     (Actif / « Comblé N % » / Mitigé). A mitigated/filled card gets `.mitig`.
 *   · `.tl` — the lifecycle timeline (only engine-recorded events; `mitigated_at`
 *     is the SINGLE FIRST contact — never a « ×N »: the engine tracks no per-test
 *     history).
 *   · `.fillbar` — FVG partial fill, from the real `fill_level` price only.
 *   · `.znarr` — a factual sentence assembled from the same engine facts.
 *   · `.zfoot` — « Analyser » (deep-link carrying the REAL engine id) and
 *     « Masquer / Afficher » (hide-by-id through the shared view state).
 *
 * The chevron unfolds ONLY the cross-timeframe geometric overlaps (a secondary
 * fact); the lifecycle itself is always visible, as in the reference.
 */

function statusLabel(zone: ZoneLifecycle, fmt: ReadingFmt): string {
  return zone.kind === 'ob'
    ? fmt.obStatus(zone.status as OBStatus)
    : fmt.fvgStatus(zone.status as FVGStatus);
}

/** `.tagx` tone by direction — bull / bear / neutral (never a decorative hue). */
function tagTone(direction: ZoneLifecycle['direction']): 'bull' | 'bear' | 'neu' {
  if (direction === 'bullish') return 'bull';
  if (direction === 'bearish') return 'bear';
  return 'neu';
}

/** "OB ↑" / "FVG ↓" — kind + a direction arrow (nothing when direction is null). */
function tagLabel(zone: ZoneLifecycle): string {
  const kind = zone.kind === 'ob' ? 'OB' : 'FVG';
  const arrow = zone.direction === 'bullish' ? ' ↑' : zone.direction === 'bearish' ? ' ↓' : '';
  return `${kind}${arrow}`;
}

/**
 * The `.zstate` badge — the zone's present lifecycle state, one of three tones:
 *   · Actif (`zs-a`)          — OB/FVG still active
 *   · « Comblé N % » (`zs-f`) — a partially-filled FVG (fraction from fill_level)
 *   · Mitigé (`zs-m`)         — a mitigated OB / fully-filled FVG (card = .mitig)
 * Descriptive only: it states what the engine records, never a future effect.
 */
function zoneState(
  zone: ZoneLifecycle,
  frac: number | null,
  t: ZonesT,
): { cls: 'zs-a' | 'zs-f' | 'zs-m'; label: string; mitig: boolean } {
  if (zone.isMitigated && zone.status === 'partially_filled' && frac != null) {
    return { cls: 'zs-f', label: t('state.filledPct', { pct: Math.round(frac * 100) }), mitig: false };
  }
  if (zone.isMitigated) {
    return { cls: 'zs-m', label: t('state.mitigated'), mitig: true };
  }
  return { cls: 'zs-a', label: t('state.active'), mitig: false };
}

/**
 * Present-tense « still there? » fact — NOT a prediction and independent of
 * whether the zone was tested: an OB is « invalidée » only once price CLOSED
 * through it, an FVG « comblée » only once fully filled.
 */
function effectiveness(zone: ZoneLifecycle, t: ZonesT): { effective: boolean; label: string } {
  if (zone.kind === 'ob') {
    return zone.status === 'invalidated'
      ? { effective: false, label: t('effectiveness.obSpentLabel') }
      : { effective: true, label: t('effectiveness.obHeldLabel') };
  }
  return zone.status === 'filled'
    ? { effective: false, label: t('effectiveness.fvgSpentLabel') }
    : { effective: true, label: t('effectiveness.fvgHeldLabel') };
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
 * reaches back to the formation, exact duration alone otherwise. Never estimated.
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
  /**
   * True when this card is the target of a `?zone=<id>` deep-link — it then
   * carries the `.zsel` highlight ring so the referenced card stands out.
   */
  isSelected?: boolean;
  /** Ref to the card root, so the workspace can scroll a deep-linked card into view. */
  cardRef?: React.Ref<HTMLElement>;
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
  isSelected = false,
  cardRef,
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
    // The live/ongoing step reads « Maintenant » (reference), not the panel label.
    active: t('timeline.currentStep'),
  };
  const durationLabels: DurationLabels = {
    underMinute: t('duration.underMinute'),
    min: t('duration.min'),
    hour: t('duration.hour'),
    day: t('duration.day'),
  };

  const events = buildTimeline(zone, timelineLabels);
  const frac = zone.status === 'partially_filled' ? fillFraction(zone) : null;

  const rel = priceRelation(zone, referencePrice);
  const relation = relationLabel(rel, instrument, t, fmt);
  const age = ageLabel(zone, candles, new Date(), t, durationLabels);
  const tested = testedLabel(zone, t);
  const overlaps = findOverlaps(zone, siblingZones);
  const state = zoneState(zone, frac, t);
  const eff = effectiveness(zone, t);

  // The live "Maintenant" sub-line: « prix dedans » when the price sits inside the
  // band, otherwise the current engine status word (both present-tense facts).
  const nowSub = rel?.position === 'inside' ? t('timeline.priceInside') : statusLabel(zone, fmt);

  // Secondary `.znarr` facts (age·tested / fill / relation), each present only
  // when its engine backing exists — joined by « · » with no dangling separator.
  const secondaryFacts = [
    tested ?? undefined,
    frac != null
      ? t('fill.label', { price: fmt.price(zone.fillLevel as number, instrument), pct: Math.round(frac * 100) })
      : undefined,
    relation ?? undefined,
  ].filter((s): s is string => Boolean(s));

  return (
    <article
      ref={cardRef}
      className={cn('zone', state.mitig && 'mitig', isHidden && 'opacity-60', isSelected && 'zsel')}
    >
      {/* Header: type/direction tag · price band · state badge */}
      <div className="ztop">
        <span className={cn('tagx', tagTone(zone.direction))}>{tagLabel(zone)}</span>
        <span className="rng">{fmt.band(zone.levelLow, zone.levelHigh, instrument)}</span>
        <span className="hsp" />
        <span className={cn('zstate', state.cls)}>{state.label}</span>
      </div>

      {/* Lifecycle timeline — only engine-recorded events (single « Testé »). */}
      <ZoneTimeline events={events} nowSubLabel={nowSub} />

      {/* FVG partial fill — from the real fill_level price only. */}
      {frac != null && (
        <div
          className="fillbar"
          role="progressbar"
          aria-valuenow={Math.round(frac * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={t('fill.aria')}
        >
          <span style={{ width: `${Math.round(frac * 100)}%` }} />
        </div>
      )}

      {/* Factual narration — assembled from the same engine facts, no prediction. */}
      <p className="znarr">
        {age && <b>{age}</b>}
        {secondaryFacts.length > 0 ? `${age ? ' · ' : ''}${secondaryFacts.join(' · ')}` : ''}
        {age || secondaryFacts.length > 0 ? ' · ' : ''}
        <b>{eff.label}</b>.
      </p>

      {/* Cross-timeframe geometric overlaps (secondary fact), unfolded on demand. */}
      {overlaps.length > 0 && (
        <>
          <button
            type="button"
            onClick={() => setExpanded((e) => !e)}
            aria-expanded={expanded}
            aria-controls={detailsId}
            className="mb-2 flex items-center gap-1 self-start text-[10.5px] font-medium text-[var(--dim)] transition-colors hover:text-[var(--txt)]"
          >
            <ChevronDown
              aria-hidden
              className={cn('h-3 w-3 transition-transform', expanded && 'rotate-180')}
            />
            {expanded ? t('details.collapse') : t('details.expand')}
          </button>
          {expanded && (
            <div id={detailsId} className="mb-2.5 flex flex-col gap-1">
              <span className="text-[9px] font-semibold uppercase tracking-wide text-[var(--faint)]">
                {t('overlaps.heading')}
              </span>
              <p className="text-[10.5px] leading-snug text-[var(--faint)]">
                {t('overlaps.description')}
              </p>
              <ul className="flex flex-col gap-0.5">
                {overlaps.map((o) => (
                  // Sibling ids are only unique WITHIN a timeframe — prefix it.
                  <li key={`${o.timeframe}-${o.id}`} className="text-[11px] text-[var(--dim)]">
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
        </>
      )}

      {/* Actions */}
      <div className="zfoot">
        <Link href={appHref} className="btn acc" aria-label={t('actions.analyzeAria')}>
          {t('actions.analyze')}
        </Link>
        <button
          type="button"
          className="btn"
          onClick={() => onToggleHide(zone.id)}
          aria-pressed={isHidden}
        >
          {isHidden ? t('actions.show') : t('actions.hide')}
        </button>
      </div>
    </article>
  );
}
