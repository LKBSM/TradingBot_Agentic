'use client';

import * as React from 'react';
import { useLocale, useTranslations } from 'next-intl';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import type { Candle, LiquidityPool } from '@/types/market-reading';
import {
  barsSince,
  buildContactTimeline,
  contactCount,
  formatDurationShort,
  formatZoneShortTime,
  fvgContactFills,
  isConsumed,
  type ContactTimelineLabels,
  type DurationLabels,
  type SiblingZone,
  type ZoneLifecycle,
  zoneHeaderState,
  zoneProximity,
} from '@/lib/zones/lifecycle';
import { buildConfluence, type ConfluenceFact } from '@/lib/zones/confluence';
import { formationSession } from '@/lib/zones/formation-session';
import { ZoneTimeline } from './ZoneTimeline';

type ZonesT = ReturnType<typeof useTranslations>;
type ReadingFmt = ReturnType<typeof useReadingFormatters>;

/**
 * One zone card (VZ-1). Every line is a present/past FACT read off the engine
 * payload — the zone's geometry, its per-contact ledger, the break it precedes.
 * The card carries, in order: header · proximity · confluence · contacts ·
 * timeline · fill · origin · details · actions. Selecting it (a click) makes it
 * the M.I.A subject — there is deliberately NO « Analyser » button.
 *
 * Vocabulary discipline (mission §0): containment is « à l'intérieur de / englobe
 * / au même niveau » (never « chevauche »); nothing is « respectée / validée /
 * solide / fiable / de qualité » — only entered / exited / traversed, with dates.
 */

function tagTone(direction: ZoneLifecycle['direction']): 'bull' | 'bear' | 'neu' {
  if (direction === 'bullish') return 'bull';
  if (direction === 'bearish') return 'bear';
  return 'neu';
}

function tagLabel(zone: ZoneLifecycle): string {
  const kind = zone.kind === 'ob' ? 'OB' : 'FVG';
  const arrow = zone.direction === 'bullish' ? ' ↑' : zone.direction === 'bearish' ? ' ↓' : '';
  return `${kind}${arrow}`;
}

/** The header badge — a fact, never a judgement. */
function headerBadge(zone: ZoneLifecycle, t: ZonesT): { cls: string; label: string } {
  const s = zoneHeaderState(zone);
  switch (s.key) {
    case 'consumed':
      return { cls: 'zs-m', label: t('header.consumed') };
    case 'contacts':
      return { cls: 'zs-f', label: t('header.contacts', { count: s.count }) };
    case 'never_filled':
      return { cls: 'zs-a', label: t('header.neverFilled') };
    default:
      return { cls: 'zs-a', label: t('header.untouched') };
  }
}

/** Full kind + direction, e.g. « Order Block haussier ». */
function kindDir(kind: 'ob' | 'fvg', dir: ZoneLifecycle['direction'], t: ZonesT, fmt: ReadingFmt): string {
  const k = kind === 'ob' ? t('kind.ob') : t('kind.fvg');
  return dir ? `${k} ${fmt.direction(dir)}` : k;
}

function edgeWord(edge: 'low' | 'high', t: ZonesT): string {
  return t(`proximity.edge.${edge}`);
}

// ─── Proximity block ─────────────────────────────────────────────────────────

function ProximityBlock({
  zone,
  price,
  instrument,
  t,
  fmt,
  locale,
}: {
  zone: ZoneLifecycle;
  price: number | null;
  instrument: string;
  t: ZonesT;
  fmt: ReadingFmt;
  locale: string;
}) {
  const prox = zoneProximity(zone, price);
  if (!prox) return null;

  const last = zone.contacts[zone.contacts.length - 1] ?? null;
  const nearEdge: 'low' | 'high' = !prox.inside && prox.side === 'above' ? 'low' : 'high';

  // « Dernier contact » line — describes the last observed contact by outcome.
  let lastLine: React.ReactNode = null;
  if (prox.inside) {
    lastLine = null; // covered by the "entré" row below
  } else if (!last) {
    lastLine = <span className="v faint">{t('proximity.neverReturned')}</span>;
  } else {
    const when = fmt.relativePast(last.at);
    const lvl = fmt.price(last.level, instrument);
    if (last.outcome === 'traversal') {
      lastLine = <span className="v">{t('proximity.lastTraversal', { when, level: lvl })}</span>;
    } else if (last.outcome === 'edge_touch') {
      lastLine = (
        <span className="v">{t('proximity.lastEdge', { when, edge: edgeWord(nearEdge, t) })}</span>
      );
    } else {
      lastLine = (
        <span className="v">
          {t('proximity.lastEntry', { when, level: lvl, exit: t(`proximity.exitBy.${nearEdge}`) })}
        </span>
      );
    }
  }

  // Gauge extent spans the band and the price, with light padding.
  const lo = Math.min(zone.levelLow, price ?? zone.levelLow);
  const hi = Math.max(zone.levelHigh, price ?? zone.levelHigh);
  const pad = Math.max((hi - lo) * 0.18, (zone.levelHigh - zone.levelLow) || 1) * 0.5;
  const eLo = lo - pad;
  const eHi = hi + pad;
  const span = eHi - eLo || 1;
  const bandLeft = ((zone.levelLow - eLo) / span) * 100;
  const bandRight = 100 - ((zone.levelHigh - eLo) / span) * 100;
  const priceLeft = price != null ? ((price - eLo) / span) * 100 : null;

  const enteredAt = prox.inside
    ? [...zone.contacts].reverse().find((c) => c.outcome === 'inside' || c.outcome === 'entry_exit')
    : null;

  return (
    <div className={cn('zpx', prox.inside && 'inside')}>
      {prox.inside ? (
        <div className="zpxr">
          <span className="k">{t('proximity.positionLabel')}</span>
          <span className="v">
            {t('proximity.insideDist', {
              low: fmt.points(prox.distToLow, instrument),
              high: fmt.points(prox.distToHigh, instrument),
            })}
          </span>
        </div>
      ) : (
        <div className="zpxr">
          <span className="k">{t('proximity.distanceLabel')}</span>
          <span className="v">
            {t('proximity.distanceLine', {
              pts: fmt.points(prox.distance, instrument),
              pct: fmt.pctShort(prox.distancePct),
              side: t(`proximity.side.${prox.side}`),
              edge: edgeWord(prox.edge, t),
            })}
          </span>
        </div>
      )}

      {prox.inside && enteredAt && (
        <div className="zpxr">
          <span className="k">{t('proximity.enteredLabel')}</span>
          <span className="v">
            {t('proximity.enteredAt', {
              when: fmt.relativePast(enteredAt.at),
              time: formatZoneShortTime(enteredAt.at, locale),
            })}
          </span>
        </div>
      )}

      {lastLine && (
        <div className="zpxr">
          <span className="k">{t('proximity.lastContactLabel')}</span>
          {lastLine}
        </div>
      )}

      {priceLeft != null && (
        <div className="zgauge" aria-hidden>
          <span className="gband" style={{ left: `${bandLeft}%`, right: `${bandRight}%` }} />
          <span
            className="gpx"
            style={{ left: `${Math.max(0, Math.min(100, priceLeft))}%` }}
          />
        </div>
      )}
    </div>
  );
}

// ─── Confluence block ────────────────────────────────────────────────────────

function confluenceLine(f: ConfluenceFact, zone: ZoneLifecycle, t: ZonesT, fmt: ReadingFmt, instrument: string): string {
  if (f.relation === 'liquidity') {
    const side = t(`liquiditySide.${f.liquiditySide}`);
    const status = fmt.liquidityStatus(f.liquidityStatus as never).label.toLowerCase();
    const level = fmt.price(f.level as number, instrument);
    if (f.distanceSide === 'inside') {
      return t('confluence.liquidityInside', { side, status, level });
    }
    return t('confluence.liquidity', {
      side,
      status,
      level,
      pts: fmt.points(f.distance as number, instrument),
      dside: t(`proximity.side.${f.distanceSide}`),
      edge: edgeWord(f.distanceSide === 'above' ? 'high' : 'low', t),
    });
  }
  const kd = kindDir(f.kind as 'ob' | 'fvg', f.direction ?? null, t, fmt);
  const band = fmt.band(f.levelLow as number, f.levelHigh as number, instrument);
  const sameUnit = f.timeframe == null;
  const tf = f.timeframe ?? '';
  if (f.relation === 'inner') {
    return sameUnit
      ? t('confluence.innerSame', { kd, band })
      : t('confluence.innerTf', { kd, tf, band });
  }
  if (f.relation === 'outer') {
    return sameUnit
      ? t('confluence.outerSame', { kd, band })
      : t('confluence.outerTf', { kd, tf, band });
  }
  return sameUnit
    ? t('confluence.sameLevelSame', { kd, band })
    : t('confluence.sameLevelTf', { kd, tf, band });
}

function ConfluenceBlock({
  facts,
  zone,
  t,
  fmt,
  instrument,
}: {
  facts: ConfluenceFact[];
  zone: ZoneLifecycle;
  t: ZonesT;
  fmt: ReadingFmt;
  instrument: string;
}) {
  const none = facts.length === 0;
  return (
    <div className={cn('zconf', none && 'none')}>
      <div className="ct">{t('confluence.heading')}</div>
      {none ? (
        <div className="cl">{t('confluence.none')}</div>
      ) : (
        facts.map((f, i) => (
          <div className="cl" key={i}>
            {confluenceLine(f, zone, t, fmt, instrument)}
          </div>
        ))
      )}
    </div>
  );
}

// ─── Contacts ledger ─────────────────────────────────────────────────────────

function ContactsBlock({
  zone,
  t,
  fmt,
  instrument,
  locale,
}: {
  zone: ZoneLifecycle;
  t: ZonesT;
  fmt: ReadingFmt;
  instrument: string;
  locale: string;
}) {
  const fills = fvgContactFills(zone);
  const rows = zone.contacts
    .map((c, i) => ({ c, i }))
    .filter(({ c }) => c.outcome !== 'inside');
  if (rows.length === 0) return null;

  return (
    <div className="zcont">
      <div className="ct">{t('contacts.heading')}</div>
      {rows.map(({ c, i }) => {
        const time = formatZoneShortTime(c.at, locale);
        const level = fmt.price(c.level, instrument);
        let text: string;
        if (c.outcome === 'edge_touch') {
          text = t('contacts.edgeTouch');
        } else if (c.outcome === 'traversal') {
          text = t('contacts.traversal', { level });
        } else {
          text = t('contacts.entryExit', { level });
        }
        const fill = zone.kind === 'fvg' && fills[i] != null ? fills[i]! : null;
        return (
          <div className="cr" key={i}>
            <i>{time}</i>
            <span>
              {text}
              {fill != null && ` ${t('contacts.fillProgress', { pct: fmt.pctShort(fill) })}`}
            </span>
          </div>
        );
      })}
      <div className="cr honesty">
        <i />
        <span>{t('contacts.honesty')}</span>
      </div>
    </div>
  );
}

// ─── Card ────────────────────────────────────────────────────────────────────

export interface ZoneLifecycleCardProps {
  zone: ZoneLifecycle;
  instrument: string;
  referencePrice: number | null;
  candles: Candle[] | null;
  /** Other LIVE zones of the same reading (for the nested-confluence facts). */
  sameTfZones: readonly ZoneLifecycle[];
  /** Zones of the other timeframes (same instrument) for the confluence facts. */
  siblingZones: readonly SiblingZone[];
  /** The reading's liquidity pockets (for the nearby-liquidity fact). */
  liquidityPools: readonly LiquidityPool[];
  isHidden: boolean;
  onToggleHide(zoneId: string): void;
  /** Focus this zone on the chart (`/app`) via its real engine id. */
  onShowOnChart(zoneId: string): void;
  /** Select this zone as the M.I.A subject (a click anywhere on the card). */
  onSelect(zoneId: string): void;
  isSelected?: boolean;
  cardRef?: React.Ref<HTMLElement>;
}

export function ZoneLifecycleCard({
  zone,
  instrument,
  referencePrice,
  candles,
  sameTfZones,
  siblingZones,
  liquidityPools,
  isHidden,
  onToggleHide,
  onShowOnChart,
  onSelect,
  isSelected = false,
  cardRef,
}: ZoneLifecycleCardProps) {
  const t = useTranslations('zones');
  const fmt = useReadingFormatters();
  const locale = useLocale();
  const [expanded, setExpanded] = React.useState(false);
  const detailsId = React.useId();

  const badge = headerBadge(zone, t);
  const height = zone.levelHigh - zone.levelLow;
  const heightLabel =
    referencePrice != null && referencePrice > 0
      ? t('card.height', {
          pts: fmt.points(height, instrument),
          pct: fmt.pctShort(height / referencePrice),
        })
      : t('card.heightPtsOnly', { pts: fmt.points(height, instrument) });

  const confluence = React.useMemo(
    () => buildConfluence(zone, sameTfZones, siblingZones, liquidityPools),
    [zone, sameTfZones, siblingZones, liquidityPools],
  );

  const timelineLabels: ContactTimelineLabels = {
    formed: t('timeline.formed'),
    entry: (n, total) => (total > 1 ? t('timeline.entryN', { n }) : t('timeline.entry')),
    touch: t('timeline.touch'),
    traversal: t('timeline.traversal'),
    now: t('timeline.now'),
  };
  const durationLabels: DurationLabels = {
    underMinute: t('duration.underMinute'),
    min: t('duration.min'),
    hour: t('duration.hour'),
    day: t('duration.day'),
  };
  const events = buildContactTimeline(zone, timelineLabels);

  // Age (details): exact duration + real bar count when the window reaches back.
  const createdMs = Date.parse(zone.createdAt);
  const ageDuration = Number.isNaN(createdMs)
    ? null
    : formatDurationShort(Math.max(Date.now() - createdMs, 0), durationLabels);
  const ageBars = barsSince(candles, zone.createdAt);

  // Canonical session from the payload; the client mirror is only a fallback.
  const session = zone.session ?? formationSession(zone.createdAt, instrument);
  const fvgFill = zone.kind === 'fvg' && zone.status === 'partially_filled'
    ? fvgContactFills(zone).filter((f): f is number => f != null).pop() ?? null
    : null;

  // ── Origin sentence (« Ce qui l'a créée ») ──
  let originLine: string | null = null;
  if (zone.kind === 'ob' && zone.origin) {
    originLine = t('origin.ob', {
      kind: zone.origin.kind === 'choch' ? 'CHOCH' : 'BOS',
      dir: fmt.direction(zone.origin.direction),
      time: formatZoneShortTime(zone.origin.at, locale),
      level: fmt.price(zone.origin.level, instrument),
    });
  } else if (zone.kind === 'fvg') {
    originLine = t('origin.fvg');
  }

  const stop = (e: React.MouseEvent) => e.stopPropagation();

  return (
    <article
      ref={cardRef}
      data-zone-id={zone.id}
      className={cn('zone', isConsumed(zone) && 'mitig', isHidden && 'zhidden', isSelected && 'zsel')}
      onClick={() => onSelect(zone.id)}
      aria-current={isSelected ? 'true' : undefined}
    >
      {/* Header */}
      <div className="ztop">
        <span className={cn('tagx', tagTone(zone.direction))}>{tagLabel(zone)}</span>
        <span className="rng">{fmt.band(zone.levelLow, zone.levelHigh, instrument)}</span>
        <span className="zhgt">{heightLabel}</span>
        <span className="hsp" />
        <span className={cn('zstate', badge.cls)}>{badge.label}</span>
      </div>

      {/* Proximity */}
      <ProximityBlock
        zone={zone}
        price={referencePrice}
        instrument={instrument}
        t={t}
        fmt={fmt}
        locale={locale}
      />

      {/* Confluence — always present (absence state when nothing detected). */}
      <ConfluenceBlock facts={confluence} zone={zone} t={t} fmt={fmt} instrument={instrument} />

      {/* Contacts ledger */}
      <ContactsBlock zone={zone} t={t} fmt={fmt} instrument={instrument} locale={locale} />

      {/* Frise de vie */}
      <ZoneTimeline events={events} nowSubLabel={t('timeline.now')} />

      {/* FVG partial fill bar */}
      {fvgFill != null && (
        <div
          className="fillbar"
          role="progressbar"
          aria-valuenow={Math.round(fvgFill * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={t('fill.aria')}
        >
          <span style={{ width: `${Math.round(fvgFill * 100)}%` }} />
        </div>
      )}

      {/* Origin */}
      {originLine && (
        <div className="zorg">
          <div className="ot">{t('origin.heading')}</div>
          <div className="ol">
            {originLine}
            {session && ` ${t('origin.session', { session: t(`session.${session}`) })}`}
          </div>
        </div>
      )}

      {/* Details (collapsed) */}
      <div className="zdet">
        <button
          type="button"
          className="zdeth"
          aria-expanded={expanded}
          aria-controls={detailsId}
          onClick={(e) => {
            stop(e);
            setExpanded((v) => !v);
          }}
        >
          <ChevronDown aria-hidden className={cn('h-3 w-3 transition-transform', expanded && 'rotate-180')} />
          {t('details.heading')}
        </button>
        {expanded && (
          <div id={detailsId} className="zkv">
            <Kv k={t('details.high')} v={fmt.price(zone.levelHigh, instrument)} mono />
            <Kv k={t('details.low')} v={fmt.price(zone.levelLow, instrument)} mono />
            <Kv k={t('details.mid')} v={fmt.price((zone.levelHigh + zone.levelLow) / 2, instrument)} mono />
            <Kv k={t('details.height')} v={heightLabel} mono />
            {ageDuration && (
              <Kv
                k={t('details.age')}
                v={ageBars != null ? t('details.ageWithBars', { duration: ageDuration, bars: ageBars }) : ageDuration}
              />
            )}
            <Kv k={t('details.entries')} v={String(zone.contacts.filter((c) => c.outcome === 'entry_exit').length)} />
            <Kv k={t('details.edgeTouches')} v={String(zone.contacts.filter((c) => c.outcome === 'edge_touch').length)} />
            {session && <Kv k={t('details.session')} v={t(`session.${session}`)} />}
            <Kv k={t('details.id')} v={zone.id} mono />
          </div>
        )}
      </div>

      {/* Actions — no « Analyser » (selecting the card is enough). */}
      <div className="zfoot">
        <button
          type="button"
          className="btn"
          onClick={(e) => {
            stop(e);
            onShowOnChart(zone.id);
          }}
        >
          {t('actions.show')}
        </button>
        <button
          type="button"
          className="btn gh"
          aria-pressed={isHidden}
          onClick={(e) => {
            stop(e);
            onToggleHide(zone.id);
          }}
        >
          {isHidden ? t('actions.unhide') : t('actions.hide')}
        </button>
      </div>
    </article>
  );
}

function Kv({ k, v, mono }: { k: string; v: string; mono?: boolean }) {
  return (
    <div className="zkvc">
      <div className="k">{k}</div>
      <div className={cn('v', mono && 'mono')}>{v}</div>
    </div>
  );
}
