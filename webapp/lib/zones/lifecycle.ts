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

import type {
  Candle,
  Direction,
  FairValueGap,
  MarketReadingStructure,
  OBImportance,
  OrderBlock,
} from '@/types/market-reading';

export type ZoneKind = 'ob' | 'fvg';

/** Lifecycle filter buckets exposed on the page. */
export type ZoneFilter = 'all' | 'active' | 'mitigated';

/**
 * Sort order for the zone list. Deliberately NO quality/importance sort: ranking
 * zones by "strength" would be an implicit recommendation (mission §0). The
 * three orders are factual: distance to price, formation recency, engine status.
 */
export type ZoneSort = 'proximity' | 'recency' | 'state';

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
 * Localized labels for the timeline steps, INJECTED by the React caller (this
 * module is a pure, hook-free lib). `obTested` / `fvgTested` disambiguate the
 * single "first contact" interaction step ("Testé" for an OB, "Pénétré" for an
 * FVG) — the engine tracks no per-test history, so there is never a "×N".
 */
export interface TimelineLabels {
  formed: string;
  mitigated: string;
  obTested: string;
  fvgTested: string;
  filled: string;
  partial: string;
  active: string;
}

/**
 * Build the lifecycle timeline from ONLY the events the engine actually tracked.
 * `Formé` is always present (every zone has `created_at`); the rest depend on
 * `tested` / `status` / `mitigated_at`. No event is emitted without a backing
 * fact, and no "×N" count is ever produced (the engine has none). Step labels
 * are supplied by the caller (`labels`) so the module stays locale-agnostic.
 */
export function buildTimeline(zone: ZoneLifecycle, labels: TimelineLabels): TimelineEvent[] {
  const events: TimelineEvent[] = [
    { key: 'formed', label: labels.formed, at: zone.createdAt, variant: 'formed' },
  ];

  if (zone.kind === 'ob') {
    if (zone.status === 'mitigated') {
      events.push({
        key: 'mitigated',
        label: labels.mitigated,
        at: zone.mitigatedAt,
        variant: 'terminal',
      });
    } else {
      if (zone.tested) {
        events.push({
          key: 'tested',
          label: labels.obTested,
          at: zone.mitigatedAt,
          variant: 'interaction',
        });
      }
      events.push({ key: 'active', label: labels.active, at: null, variant: 'ongoing' });
    }
    return events;
  }

  // FVG
  if (zone.status === 'filled') {
    if (zone.tested && zone.mitigatedAt) {
      events.push({
        key: 'tested',
        label: labels.fvgTested,
        at: zone.mitigatedAt,
        variant: 'interaction',
      });
    }
    // The engine records no "filled_at" — surface the terminal step without a date.
    events.push({ key: 'filled', label: labels.filled, at: null, variant: 'terminal' });
    return events;
  }

  if (zone.status === 'partially_filled') {
    events.push({
      key: 'partial',
      label: labels.partial,
      at: zone.mitigatedAt,
      variant: 'interaction',
    });
    events.push({ key: 'active', label: labels.active, at: null, variant: 'ongoing' });
    return events;
  }

  // active
  if (zone.tested && zone.mitigatedAt) {
    events.push({
      key: 'tested',
      label: labels.fvgTested,
      at: zone.mitigatedAt,
      variant: 'interaction',
    });
  }
  events.push({ key: 'active', label: labels.active, at: null, variant: 'ongoing' });
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

// ─── Present-tense facts (price relation · age · overlaps) ───────────────────

/**
 * Where the zone's band sits RELATIVE TO the current price — a present-tense
 * geometric fact, recomputed at the freshness of the readings. `inside` when the
 * price is within [levelLow, levelHigh]; otherwise the side the ZONE is on
 * (`above`/`below` the price) plus the gap to its NEAREST edge, in price units.
 * Returns null without a usable price — the badge is then omitted, never guessed.
 */
export type PriceRelation =
  | { position: 'inside' }
  | { position: 'above' | 'below'; distance: number };

export function priceRelation(
  zone: ZoneLifecycle,
  price: number | null | undefined,
): PriceRelation | null {
  if (price == null || !Number.isFinite(price)) return null;
  if (price >= zone.levelLow && price <= zone.levelHigh) return { position: 'inside' };
  return price > zone.levelHigh
    ? { position: 'below', distance: price - zone.levelHigh }
    : { position: 'above', distance: zone.levelLow - price };
}

/**
 * Number of candles CLOSED strictly after the zone's formation bar, counted on
 * the real candle window (ascending) — never derived by dividing elapsed time by
 * the timeframe (weekends/session gaps would inflate it). Returns null when the
 * window does not reach back to the formation (count would be truncated), when
 * no candles are available, or on an unparsable timestamp — the card then falls
 * back to the exact duration alone. Honest degradation, no estimate.
 */
export function barsSince(
  candles: Candle[] | null | undefined,
  createdAtIso: string,
): number | null {
  if (!candles || candles.length === 0) return null;
  const ms = Date.parse(createdAtIso);
  if (Number.isNaN(ms)) return null;
  const sec = ms / 1000;
  if (sec < candles[0]!.time) return null; // formation predates the window
  let count = 0;
  for (let i = candles.length - 1; i >= 0 && candles[i]!.time > sec; i -= 1) count += 1;
  return count;
}

/**
 * Localized unit fragments for {@link formatDurationShort}, INJECTED by the
 * React caller (hook-free lib). `underMinute` is the whole "moins d'une minute"
 * string; `min` / `hour` / `day` are the compact unit suffixes ("min" / "h" /
 * "j" in French).
 */
export interface DurationLabels {
  underMinute: string;
  min: string;
  hour: string;
  day: string;
}

/** "45 min", "6 h 30", "2 j 4 h" — compact elapsed duration (fact, no rounding up). */
export function formatDurationShort(ms: number, labels: DurationLabels): string {
  const min = Math.floor(ms / 60_000);
  if (min < 1) return labels.underMinute;
  if (min < 60) return `${min} ${labels.min}`;
  const h = Math.floor(min / 60);
  const m = min % 60;
  if (h < 24) return m === 0 ? `${h} ${labels.hour}` : `${h} ${labels.hour} ${String(m).padStart(2, '0')}`;
  const d = Math.floor(h / 24);
  const rh = h % 24;
  return rh === 0 ? `${d} ${labels.day}` : `${d} ${labels.day} ${rh} ${labels.hour}`;
}

/**
 * A zone from ANOTHER timeframe of the same instrument, reduced to what the
 * geometric-overlap fact needs. Built from the sibling readings' engine output
 * (`collectZones`) — same source of truth as the cards themselves.
 */
export interface SiblingZone {
  id: string;
  kind: ZoneKind;
  direction: Direction | null;
  levelHigh: number;
  levelLow: number;
  /** Timeframe the sibling zone was read on (e.g. 'H1'). */
  timeframe: string;
}

/**
 * Pure interval intersection between the zone's band and each sibling's band
 * (strict: touching edges alone don't count). A geometric FACT — the caller must
 * phrase it as such ("chevauche un OB H1 (bornes)"), never as confluence,
 * reinforcement or reliability (mission §0).
 */
export function findOverlaps(
  zone: ZoneLifecycle,
  siblings: readonly SiblingZone[],
): SiblingZone[] {
  return siblings.filter(
    (s) => s.levelLow < zone.levelHigh && zone.levelLow < s.levelHigh,
  );
}

// ─── Date formatting ─────────────────────────────────────────────────────────

/** "28 juin 2026" — absolute date, locale-aware (defaults to fr-FR). */
export function formatZoneDate(iso: string, locale: string = 'fr-FR'): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat(locale, { dateStyle: 'medium' }).format(d);
}

/** "28 juin 2026, 14:30" — date + time for the timeline steps, locale-aware. */
export function formatZoneDateTime(iso: string, locale: string = 'fr-FR'): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat(locale, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(d);
}

// ─── Filter + sort (display-only) ────────────────────────────────────────────

export function matchesFilter(zone: ZoneLifecycle, filter: ZoneFilter): boolean {
  if (filter === 'all') return true;
  if (filter === 'active') return zone.isActive;
  return zone.isMitigated;
}

// Engine-status order (factual lifecycle progression, NOT a quality ranking).
const STATE_RANK: Record<string, number> = {
  active: 0,
  partially_filled: 1,
  mitigated: 2,
  filled: 3,
  invalidated: 4,
};

export function stateRank(zone: ZoneLifecycle): number {
  return STATE_RANK[zone.status] ?? 5;
}

/**
 * Distance from the price to the zone's BAND (0 when inside) — the same fact the
 * card's relation badge shows, so the proximity order matches what is displayed.
 */
function distanceToPrice(zone: ZoneLifecycle, price: number): number {
  const rel = priceRelation(zone, price);
  return rel && rel.position !== 'inside' ? rel.distance : 0;
}

/**
 * Order the zones for display. `proximity` (the default) uses the distance from
 * the price to the band, closest first; `recency` uses `created_at` (newest
 * first); `state` follows the engine's lifecycle status. Without a usable price,
 * `proximity` degrades to the state order (no invented distance). Every order is
 * a fact — there is deliberately NO importance/quality sort (mission §0).
 */
export function sortZones(
  zones: ZoneLifecycle[],
  sort: ZoneSort,
  price?: number | null,
): ZoneLifecycle[] {
  const arr = [...zones];
  const byRecency = (a: ZoneLifecycle, b: ZoneLifecycle) =>
    new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
  if (sort === 'recency') {
    arr.sort(byRecency);
  } else if (sort === 'proximity' && price != null) {
    arr.sort(
      (a, b) => distanceToPrice(a, price) - distanceToPrice(b, price) || byRecency(a, b),
    );
  } else {
    arr.sort((a, b) => stateRank(a) - stateRank(b) || byRecency(a, b));
  }
  return arr;
}
