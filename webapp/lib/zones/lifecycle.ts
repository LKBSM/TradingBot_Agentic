/**
 * Zone lifecycle — a DISPLAY-ONLY projection of the Order Block / Fair Value Gap
 * facts the detection engine already emitted (`MarketReadingStructure`). It NEVER
 * runs detection, mutates a zone, or invents an event: every field here is read
 * straight off `OrderBlock` / `FairValueGap`, and the timeline only surfaces the
 * events the engine actually records.
 *
 * The honest boundaries of the engine's data (see the diagnostic):
 *   · `tested` is a BOOLEAN — there is NO interaction count and NO per-test
 *     history. So we say "Testé", never "Testé ×N".
 *   · `mitigated_at` is the FIRST interaction timestamp (OB mitigation point /
 *     FVG first entry) — the only lifecycle timestamp besides `created_at`.
 *   · FVG `fill_level` is the deepest penetration PRICE, not a percent. We derive
 *     a filled fraction from it geometrically (real bounds), never a forecast.
 *   · A missing timestamp degrades to "no date", never a fabricated one.
 *
 * Strictly descriptive (niveau 1.5): no target, no conviction, no prediction.
 */

import {
  formatBand,
  formatDirection,
  formatObImportance,
} from '@/lib/market-reading/formatters';
import type {
  Direction,
  FairValueGap,
  MarketReadingStructure,
  OBImportance,
  OrderBlock,
} from '@/types/market-reading';

export type ZoneKind = 'ob' | 'fvg';

/** Lifecycle filter buckets exposed on the page. */
export type ZoneFilter = 'all' | 'active' | 'mitigated';

/** Sort order for the zone list (display-only ranking). */
export type ZoneSort = 'importance' | 'recency' | 'proximity';

/**
 * A unified view of one detected zone (OB or FVG) for the Zones page. Mirrors the
 * engine fields 1:1 and adds only derived display booleans — no new facts.
 */
export interface ZoneLifecycle {
  id: string;
  kind: ZoneKind;
  direction: Direction | null;
  levelHigh: number;
  levelLow: number;
  /** OB importance bucket. `null` for FVG — the engine emits none. */
  importance: OBImportance | null;
  /** Raw engine status (`OBStatus` for OB, `FVGStatus` for FVG). */
  status: string;
  createdAt: string;
  /** Boolean only — no count is available. */
  tested: boolean;
  /** First-interaction timestamp (OB mitigation point / FVG first entry), or null. */
  mitigatedAt: string | null;
  /** FVG only — deepest penetration PRICE within the band (not a percent). */
  fillLevel: number | null;
  /** OB 'active' / FVG 'active'. */
  isActive: boolean;
  /** OB 'mitigated' / FVG 'filled' | 'partially_filled'. */
  isMitigated: boolean;
}

function obToLifecycle(ob: OrderBlock): ZoneLifecycle {
  return {
    id: ob.id,
    kind: 'ob',
    direction: ob.direction ?? null,
    levelHigh: ob.level_high,
    levelLow: ob.level_low,
    importance: ob.importance,
    status: ob.status,
    createdAt: ob.created_at,
    tested: ob.tested,
    mitigatedAt: ob.mitigated_at ?? null,
    fillLevel: null,
    isActive: ob.status === 'active',
    isMitigated: ob.status === 'mitigated',
  };
}

function fvgToLifecycle(fvg: FairValueGap): ZoneLifecycle {
  return {
    id: fvg.id,
    kind: 'fvg',
    direction: fvg.direction ?? null,
    levelHigh: fvg.level_high,
    levelLow: fvg.level_low,
    importance: null,
    status: fvg.status,
    createdAt: fvg.created_at,
    tested: fvg.tested,
    mitigatedAt: fvg.mitigated_at ?? null,
    fillLevel: fvg.fill_level ?? null,
    isActive: fvg.status === 'active',
    isMitigated: fvg.status === 'filled' || fvg.status === 'partially_filled',
  };
}

/** Project a structure's OB + FVG lists into a single zone-lifecycle list. */
export function collectZones(
  structure: MarketReadingStructure | null | undefined,
): ZoneLifecycle[] {
  if (!structure) return [];
  const obs = (structure.order_blocks ?? []).map(obToLifecycle);
  const fvgs = (structure.fair_value_gaps ?? []).map(fvgToLifecycle);
  return [...obs, ...fvgs];
}

// ─── Timeline ────────────────────────────────────────────────────────────────

/**
 * One lifecycle event. `at` is null when the engine records the event's
 * existence but not its timestamp (e.g. a fully-filled FVG has no "filled_at") —
 * we then render the step without a date rather than fabricate one.
 */
export interface TimelineEvent {
  key: 'formed' | 'tested' | 'partial' | 'mitigated' | 'filled' | 'active';
  label: string;
  at: string | null;
  variant: 'formed' | 'interaction' | 'terminal' | 'ongoing';
}

/**
 * Build the lifecycle timeline from ONLY the events the engine actually tracked.
 * `Formé` is always present (every zone has `created_at`); the rest depend on
 * `tested` / `status` / `mitigated_at`. No event is emitted without a backing
 * fact, and no "×N" count is ever produced (the engine has none).
 */
export function buildTimeline(zone: ZoneLifecycle): TimelineEvent[] {
  const events: TimelineEvent[] = [
    { key: 'formed', label: 'Formé', at: zone.createdAt, variant: 'formed' },
  ];

  if (zone.kind === 'ob') {
    if (zone.status === 'mitigated') {
      events.push({
        key: 'mitigated',
        label: 'Mitigé',
        at: zone.mitigatedAt,
        variant: 'terminal',
      });
    } else {
      if (zone.tested) {
        events.push({
          key: 'tested',
          label: 'Testé',
          at: zone.mitigatedAt,
          variant: 'interaction',
        });
      }
      events.push({ key: 'active', label: 'Suivi en cours', at: null, variant: 'ongoing' });
    }
    return events;
  }

  // FVG
  if (zone.status === 'filled') {
    if (zone.tested && zone.mitigatedAt) {
      events.push({
        key: 'tested',
        label: 'Pénétré',
        at: zone.mitigatedAt,
        variant: 'interaction',
      });
    }
    // The engine records no "filled_at" — surface the terminal step without a date.
    events.push({ key: 'filled', label: 'Comblé', at: null, variant: 'terminal' });
    return events;
  }

  if (zone.status === 'partially_filled') {
    events.push({
      key: 'partial',
      label: 'Partiellement comblé',
      at: zone.mitigatedAt,
      variant: 'interaction',
    });
    events.push({ key: 'active', label: 'Suivi en cours', at: null, variant: 'ongoing' });
    return events;
  }

  // active
  if (zone.tested && zone.mitigatedAt) {
    events.push({
      key: 'tested',
      label: 'Pénétré',
      at: zone.mitigatedAt,
      variant: 'interaction',
    });
  }
  events.push({ key: 'active', label: 'Suivi en cours', at: null, variant: 'ongoing' });
  return events;
}

// ─── Fill fraction (FVG) ─────────────────────────────────────────────────────

/**
 * Fraction of an FVG's band already penetrated, derived from the engine's
 * `fill_level` PRICE and the zone's real bounds. Direction-aware geometry: a
 * bullish gap fills downward from the top (deepest = lowest price reached →
 * filled = high − fill), a bearish gap fills upward from the bottom (deepest =
 * highest reached → filled = fill − low). Clamped to [0, 1]. Returns null when
 * not an FVG, no `fill_level`, or a degenerate band — never a guess.
 */
export function fillFraction(zone: ZoneLifecycle): number | null {
  if (zone.kind !== 'fvg' || zone.fillLevel == null) return null;
  const span = zone.levelHigh - zone.levelLow;
  if (span <= 0) return null;
  const raw =
    zone.direction === 'bearish'
      ? (zone.fillLevel - zone.levelLow) / span
      : (zone.levelHigh - zone.fillLevel) / span;
  return Math.max(0, Math.min(1, raw));
}

// ─── Narration (deterministic, factual) ──────────────────────────────────────

/** "28 juin 2026" — absolute date, no relative-to-now dependency. */
export function formatZoneDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat('fr-FR', { dateStyle: 'medium' }).format(d);
}

/** "28 juin 2026, 14:30" — date + time for the timeline steps. */
export function formatZoneDateTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat('fr-FR', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(d);
}

/**
 * One factual sentence describing the zone's life so far — present/past only,
 * composed strictly from engine facts. No target, no "va rebondir", no
 * conviction. A missing timestamp degrades the clause, never invents a date.
 */
export function narrateZone(zone: ZoneLifecycle, instrument: string): string {
  const kindLabel = zone.kind === 'ob' ? 'Order Block' : 'Fair Value Gap';
  const dir = zone.direction ? ` ${formatDirection(zone.direction)}` : '';
  const band = formatBand(zone.levelLow, zone.levelHigh, instrument);
  const parts: string[] = [
    `${kindLabel}${dir} formé le ${formatZoneDate(zone.createdAt)}, entre ${band}.`,
  ];

  if (zone.kind === 'ob' && zone.importance) {
    parts.push(`Importance ${formatObImportance(zone.importance)}.`);
  }

  if (zone.kind === 'ob') {
    if (zone.status === 'mitigated') {
      parts.push(
        zone.mitigatedAt
          ? `Mitigé le ${formatZoneDate(zone.mitigatedAt)}.`
          : 'Mitigé depuis sa formation.',
      );
    } else if (zone.tested) {
      parts.push(
        zone.mitigatedAt
          ? `Testé le ${formatZoneDate(zone.mitigatedAt)}, toujours actif.`
          : 'Déjà testé, toujours actif.',
      );
    } else {
      parts.push('Non testé à ce jour.');
    }
  } else {
    if (zone.status === 'filled') {
      parts.push('Désormais comblé.');
    } else if (zone.status === 'partially_filled') {
      const frac = fillFraction(zone);
      parts.push(
        frac != null
          ? `Partiellement comblé (≈ ${Math.round(frac * 100)} %).`
          : 'Partiellement comblé.',
      );
    } else if (zone.tested) {
      parts.push(
        zone.mitigatedAt
          ? `Pénétré le ${formatZoneDate(zone.mitigatedAt)}, encore ouvert.`
          : 'Déjà pénétré, encore ouvert.',
      );
    } else {
      parts.push('Intact.');
    }
  }

  return parts.join(' ');
}

// ─── Filter + sort (display-only) ────────────────────────────────────────────

export function matchesFilter(zone: ZoneLifecycle, filter: ZoneFilter): boolean {
  if (filter === 'all') return true;
  if (filter === 'active') return zone.isActive;
  return zone.isMitigated;
}

// Lower rank surfaces first. OB by importance; FVG (no importance) by status, so
// the two families interleave coherently — mirrors StructureSection's ordering.
const OB_IMPORTANCE_RANK: Record<OBImportance, number> = { high: 0, medium: 1, low: 2 };
const STATUS_RANK: Record<string, number> = {
  active: 0,
  partially_filled: 1,
  mitigated: 2,
  filled: 3,
  invalidated: 4,
};

export function zoneRank(zone: ZoneLifecycle): number {
  if (zone.kind === 'ob') {
    return zone.importance != null ? OB_IMPORTANCE_RANK[zone.importance] : 3;
  }
  return STATUS_RANK[zone.status] ?? 5;
}

function zoneMid(zone: ZoneLifecycle): number {
  return (zone.levelHigh + zone.levelLow) / 2;
}

/**
 * Order the zones for display. `importance` uses the rank above, `recency` uses
 * `created_at` (newest first), `proximity` uses |mid − price| (closest first,
 * falling back to importance when no price is available). Pure, stable input.
 */
export function sortZones(
  zones: ZoneLifecycle[],
  sort: ZoneSort,
  price?: number | null,
): ZoneLifecycle[] {
  const arr = [...zones];
  if (sort === 'recency') {
    arr.sort(
      (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
    );
  } else if (sort === 'proximity' && price != null) {
    arr.sort(
      (a, b) => Math.abs(zoneMid(a) - price) - Math.abs(zoneMid(b) - price),
    );
  } else {
    arr.sort((a, b) => zoneRank(a) - zoneRank(b) || zoneMid(b) - zoneMid(a));
  }
  return arr;
}
