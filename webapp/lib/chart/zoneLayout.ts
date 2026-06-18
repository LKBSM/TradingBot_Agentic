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

/** Curation caps — mechanical anti-clutter, documented (NOT importance ranking). */
export const ACTIVE_ZONE_CAP = 4;
export const TESTED_ZONE_CAP = 3;

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
 * Curate the full zone set into the boxes the chart should draw: at most
 * `ACTIVE_ZONE_CAP` active zones and `TESTED_ZONE_CAP` tested ones, chosen by
 * recency + proximity to the current price. The rest are hidden (logged by the
 * caller if useful). Active zones are returned last so they layer ON TOP of the
 * faded tested boxes when rendered in order.
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
