/**
 * Pure layout helpers for the chart's OB / FVG zone boxes.
 *
 * Split out of `ReadingChart` so the curation + time-bounding logic is unit
 * testable without a canvas / lightweight-charts instance. NOTHING here touches
 * the engine, recomputes a zone, or projects into the future — it only reads the
 * descriptive fields the backend already emits (`created_at`, `mitigated_at`,
 * `status`, price band) and decides which boxes to draw and how to bound them.
 *
 * `mitigated_at` is READ-ONLY here: it is consumed to bound a tested box at its
 * mitigation point, never written or derived.
 */
import type {
  FairValueGap,
  MarketReadingStructure,
  OrderBlock,
} from '@/types/market-reading';

/** A zone reduced to what the chart needs to place a localized, bounded box. */
export interface ZoneModel {
  id: string;
  kind: 'ob' | 'fvg';
  /** Price band (engine values, untouched). */
  high: number;
  low: number;
  /** Formation time (x-start anchor), UNIX seconds. */
  createdSec: number;
  /**
   * Mitigation point (x-end anchor for a tested zone), UNIX seconds, or null
   * while the zone is still active — an active box extends to the current bar.
   */
  mitigatedSec: number | null;
  /** Tested/mitigated (faded) vs active (crisp). */
  tested: boolean;
  label: string;
}

/**
 * Curation caps — mechanical anti-clutter, NOT importance ranking.
 *
 * ACTIVE zones are NEVER dropped by recency/proximity: an active OB/FVG can sit
 * far from the current price or be days old and still be in play (e.g. an
 * untested supply zone 50+ points away that the engine still tracks as `active`).
 * This cap is therefore a pure runaway / perf guard, set well ABOVE the backend's
 * per-type ceiling (`MAX_ZONES_PER_TYPE = 12`), so it never removes a real active
 * zone — it only bounds a pathological payload. TESTED zones (touched-but-alive:
 * `mitigated` OB / `partially_filled` FVG) stay visible and extend too; a
 * generous cap keeps the canvas calm without hiding an in-play zone in practice.
 */
export const ACTIVE_ZONE_CAP = 24;
export const TESTED_ZONE_CAP = 12;

/** ISO-8601 → UNIX seconds; returns NaN for an unparseable string. */
export function isoToSec(iso: string | null | undefined): number {
  if (!iso) return NaN;
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? NaN : Math.floor(ms / 1000);
}

/**
 * Resolve the still-OPEN price band of an FVG box. For a partially-filled gap
 * with a known direction and an in-band `fill_level`, the box shrinks to the
 * unfilled side (bullish: top down to fill_level; bearish: bottom up to it).
 * In every other case (active, unknown direction, or out-of-band level) the
 * full engine band is returned unchanged — never guess a shrink.
 */
export function openFvgBand(fvg: FairValueGap): { high: number; low: number } {
  let high = fvg.level_high;
  let low = fvg.level_low;
  const lvl = fvg.fill_level;
  if (
    fvg.status === 'partially_filled' &&
    typeof lvl === 'number' &&
    lvl > low &&
    lvl < high
  ) {
    if (fvg.direction === 'bullish') high = lvl;
    else if (fvg.direction === 'bearish') low = lvl;
  }
  return { high, low };
}

/**
 * Reduce a structure's OB/FVG zones to drawable models. Consumed zones
 * (invalidated OB, filled FVG) are skipped defensively — the backend already
 * drops them, but the chart must never surface a consumed zone as live.
 */
export function buildZoneModels(structure: MarketReadingStructure): ZoneModel[] {
  const out: ZoneModel[] = [];
  for (const ob of structure.order_blocks) {
    if (ob.status === 'invalidated') continue;
    out.push({
      id: ob.id,
      kind: 'ob',
      high: ob.level_high,
      low: ob.level_low,
      createdSec: isoToSec(ob.created_at),
      mitigatedSec: ob.mitigated_at ? isoToSec(ob.mitigated_at) : null,
      tested: ob.status !== 'active',
      label: 'Order Block',
    });
  }
  for (const fvg of structure.fair_value_gaps) {
    if (fvg.status === 'filled') continue;
    // Shrink a partially-filled gap to its STILL-OPEN portion. The engine emits
    // fill_level = the deepest wick into the band (read-only). A bullish gap
    // fills from the top down → the open part is below the penetration (high =
    // fill_level); a bearish gap fills from the bottom up → open part above it
    // (low = fill_level). Only shrink when the direction is known and the level
    // sits strictly inside the band — otherwise keep the full box (no guess).
    const { high, low } = openFvgBand(fvg);
    out.push({
      id: fvg.id,
      kind: 'fvg',
      high,
      low,
      createdSec: isoToSec(fvg.created_at),
      mitigatedSec: fvg.mitigated_at ? isoToSec(fvg.mitigated_at) : null,
      tested: fvg.status !== 'active',
      label: 'Fair Value Gap',
    });
  }
  // Drop zones with an unparseable formation time — can't be anchored.
  return out.filter((z) => Number.isFinite(z.createdSec));
}

/**
 * Where a zone box's RIGHT edge should land:
 *   · `mitigation` — a RESOLVED zone (engine `invalidated` OB / `filled` FVG, with
 *     a known mitigation point) ends AT that point (UNIX seconds). Bounded, never
 *     over-extended past a settled outcome. The engine drops resolved zones, so in
 *     practice they never reach the chart — this is a defensive guard only.
 *   · `active` — every IN-PLAY zone runs a LITTLE PAST the current bar: just beyond
 *     the latest candle, never to the plot edge (an infinite band) and never short
 *     of the candle. This covers BOTH an `active` (untested) zone AND a
 *     touched-but-alive one (`mitigated` OB / `partially_filled` FVG: price tapped
 *     it but did NOT close through, so the engine keeps it). A touched zone is
 *     still in play, so its box extends — the faded `tested` style is what marks it
 *     as already touched (symptom (b): it no longer stops at the tap).
 *
 * Pure + view-independent: the caller maps `mitigation` → x(sec) and `active` →
 * the current-bar pixel plus a small pad. Split out so the rule is unit-testable.
 */
export type ZoneRightAnchor =
  | { kind: 'mitigation'; sec: number }
  | { kind: 'active' };

export function zoneRightAnchor(zone: {
  tested: boolean;
  mitigatedSec: number | null;
  /**
   * Engine-resolved (invalidated OB / filled FVG) — a closed outcome. Drawn zones
   * are never resolved (they are filtered upstream), so this defaults to falsy and
   * the box extends; the flag only exists so the bounded branch stays expressible
   * and unit-testable.
   */
  resolved?: boolean;
}): ZoneRightAnchor {
  if (zone.resolved && zone.mitigatedSec !== null) {
    return { kind: 'mitigation', sec: zone.mitigatedSec };
  }
  return { kind: 'active' };
}

// ─── Live (provisional, intra-candle) zone interaction ────────────────────────
//
// PROTOTYPE — opt-in live overlay. Everything below derives a PROVISIONAL view
// from the latest tick price ON TOP of the candle-CONFIRMED structure, WITHOUT
// mutating it. It is recomputed from the current price each time, so when price
// retreats the provisional state cleanly disappears (nothing "confirmed" ever
// un-confirms). It only describes INTERACTION (FVG being filled, OB being
// touched) — it NEVER detects structure and NEVER touches BOS/CHOCH (those are
// candle-close only, SMC law).

/** The still-open FVG band as it stands RIGHT NOW given the live price. */
export interface LiveFvgFront {
  id: string;
  /** Provisional open-band top, intra-candle. */
  high: number;
  /** Provisional open-band bottom, intra-candle. */
  low: number;
}

/** Provisional interaction overlay computed from the latest tick. */
export interface LiveOverlay {
  /** FVGs the price is currently eating into deeper than the confirmed front. */
  fvgFronts: LiveFvgFront[];
  /** Ids of ACTIVE order blocks the price is currently inside ("in test"). */
  obInTest: Set<string>;
}

/**
 * Provisional still-open band of an FVG given the live price, or null when the
 * price isn't penetrating it any deeper than the candle-confirmed front. The
 * box shrinks toward the side price is eating from (bullish gap fills top-down →
 * new high = price; bearish fills bottom-up → new low = price). Returns null —
 * i.e. "show the confirmed box, no live shrink" — for a filled gap, an unknown
 * direction, a non-finite price, or a price outside the confirmed open band
 * (so a retreat reverts cleanly to the confirmed geometry).
 */
export function provisionalOpenFvgBand(
  fvg: FairValueGap,
  livePrice: number,
): { high: number; low: number } | null {
  if (fvg.status === 'filled') return null;
  if (!Number.isFinite(livePrice)) return null;
  const dir = fvg.direction;
  if (dir !== 'bullish' && dir !== 'bearish') return null;
  // Confirmed still-open band (already accounts for any close-confirmed fill).
  const { high: cHigh, low: cLow } = openFvgBand(fvg);
  // Price must be strictly INSIDE the confirmed open band to shrink it further.
  if (livePrice <= cLow || livePrice >= cHigh) return null;
  return dir === 'bullish'
    ? { high: livePrice, low: cLow } // eaten from the top down
    : { high: cHigh, low: livePrice }; // eaten from the bottom up
}

/**
 * True when an ACTIVE (untested) order block currently contains the live price —
 * i.e. it is being tested right now (provisional). Only an `active` OB qualifies:
 * a mitigated/invalidated OB is already a candle-confirmed outcome, never
 * "live in test".
 */
export function isObInTestLive(ob: OrderBlock, livePrice: number): boolean {
  if (ob.status !== 'active') return false;
  if (!Number.isFinite(livePrice)) return false;
  const lo = Math.min(ob.level_low, ob.level_high);
  const hi = Math.max(ob.level_low, ob.level_high);
  return livePrice >= lo && livePrice <= hi;
}

/**
 * Build the provisional interaction overlay from a structure + the live price.
 * Pure + cheap; the chart recomputes it on each new tick. A non-finite price
 * yields an empty overlay (no live state).
 */
export function buildLiveOverlay(
  structure: MarketReadingStructure,
  livePrice: number | null | undefined,
): LiveOverlay {
  const fvgFronts: LiveFvgFront[] = [];
  const obInTest = new Set<string>();
  if (livePrice == null || !Number.isFinite(livePrice)) {
    return { fvgFronts, obInTest };
  }
  for (const fvg of structure.fair_value_gaps) {
    const band = provisionalOpenFvgBand(fvg, livePrice);
    if (band) fvgFronts.push({ id: fvg.id, high: band.high, low: band.low });
  }
  for (const ob of structure.order_blocks) {
    if (isObInTestLive(ob, livePrice)) obInTest.add(ob.id);
  }
  return { fvgFronts, obInTest };
}

const mid = (z: ZoneModel) => (z.high + z.low) / 2;

/**
 * Display filter for the chatbot's `filter_zones` view action. PURELY a display
 * choice over the DETECTED zones — it hides boxes, never edits a band:
 *   · activeOnly    — drop tested/mitigated zones, keep only active ones.
 *   · minSizePct    — drop zones whose band height is < pct of the current price.
 *   · proximityOnly — keep only zones whose mid sits within ±proximityPct of price.
 * A non-finite price disables the price-relative filters (returns size-filtered
 * only). Returns a new array; the inputs are never mutated.
 */
export interface ZoneDisplayFilter {
  activeOnly: boolean;
  proximityOnly: boolean;
  proximityPct: number;
  minSizePct: number | null;
}

export function filterZoneModels(
  zones: ZoneModel[],
  currentPrice: number,
  filter: ZoneDisplayFilter,
): ZoneModel[] {
  const priceOk = Number.isFinite(currentPrice) && currentPrice > 0;
  return zones.filter((z) => {
    if (filter.activeOnly && z.tested) return false;
    if (filter.minSizePct != null && priceOk) {
      const heightPct = (Math.abs(z.high - z.low) / currentPrice) * 100;
      if (heightPct < filter.minSizePct) return false;
    }
    if (filter.proximityOnly && priceOk) {
      const distPct = (Math.abs(mid(z) - currentPrice) / currentPrice) * 100;
      if (distPct > filter.proximityPct) return false;
    }
    return true;
  });
}

/**
 * Apply the chatbot's per-id visibility state (`hide_zones` / `isolate_zones`) to
 * a zone list. PURELY a display choice over the DETECTED zones — it hides boxes
 * by id, never edits a band or invents a zone:
 *   · isolatedZoneIds — when non-null, keep ONLY zones whose id is in the set
 *     (show only the isolated zones); null means no isolation.
 *   · hiddenZoneIds   — drop any zone whose id was explicitly masked.
 * Both are reversible from the view state (`show_zones` / `reset_view`). Returns a
 * new array; the input is never mutated, and the zones themselves are untouched.
 */
export function applyZoneVisibility(
  zones: ZoneModel[],
  hiddenZoneIds: readonly string[],
  isolatedZoneIds: readonly string[] | null,
): ZoneModel[] {
  const hidden = new Set(hiddenZoneIds);
  const isolated = isolatedZoneIds === null ? null : new Set(isolatedZoneIds);
  return zones.filter((z) => {
    if (isolated !== null && !isolated.has(z.id)) return false;
    if (hidden.has(z.id)) return false;
    return true;
  });
}

/**
 * Rank a zone list by a combined RECENCY + PROXIMITY-to-price score and keep the
 * top `cap`. Mechanical curation only:
 *   · proximityRank — distance of the zone mid-price to the current price (0 = closest)
 *   · recencyRank   — formation time, most recent first (0 = newest)
 *   · score = proximityRank + recencyRank  → both weigh equally, lowest wins.
 * Ties break on id for determinism. This is NOT a measure of predictive
 * importance (that awaits annotation) — just "what's worth showing without
 * crowding the canvas".
 */
function curate(zones: ZoneModel[], currentPrice: number, cap: number): ZoneModel[] {
  if (zones.length <= cap) {
    // Still return them in the same deterministic order used when capping.
    return rankSort(zones, currentPrice);
  }
  return rankSort(zones, currentPrice).slice(0, cap);
}

function rankSort(zones: ZoneModel[], currentPrice: number): ZoneModel[] {
  const byProx = [...zones].sort((a, b) => Math.abs(mid(a) - currentPrice) - Math.abs(mid(b) - currentPrice));
  const byRec = [...zones].sort((a, b) => b.createdSec - a.createdSec);
  const proxRank = new Map(byProx.map((z, i) => [z.id, i]));
  const recRank = new Map(byRec.map((z, i) => [z.id, i]));
  const score = (z: ZoneModel) => (proxRank.get(z.id) ?? 0) + (recRank.get(z.id) ?? 0);
  return [...zones].sort((a, b) => {
    const d = score(a) - score(b);
    return d !== 0 ? d : a.id.localeCompare(b.id);
  });
}

/**
 * Curate the full zone set into the boxes the chart should draw. The split is by
 * lifecycle, NOT by recency: active zones are bounded only by `ACTIVE_ZONE_CAP` (a
 * perf guard well above the backend ceiling — an active zone is never dropped for
 * being far or old), tested-but-alive zones by the generous `TESTED_ZONE_CAP`.
 * The `rankSort` ordering (recency + proximity) only decides layout order and the
 * never-reached overflow tail. Active zones are returned last so they layer ON TOP
 * of the faded tested boxes when rendered in order.
 */
export function curateZones(
  zones: ZoneModel[],
  currentPrice: number,
  caps: { active?: number; tested?: number } = {},
): { active: ZoneModel[]; tested: ZoneModel[] } {
  const activeCap = caps.active ?? ACTIVE_ZONE_CAP;
  const testedCap = caps.tested ?? TESTED_ZONE_CAP;
  const active = curate(zones.filter((z) => !z.tested), currentPrice, activeCap);
  const tested = curate(zones.filter((z) => z.tested), currentPrice, testedCap);
  return { active, tested };
}
