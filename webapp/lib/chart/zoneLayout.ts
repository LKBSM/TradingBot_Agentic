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
import type { MarketReadingStructure } from '@/types/market-reading';

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
    out.push({
      id: fvg.id,
      kind: 'fvg',
      high: fvg.level_high,
      low: fvg.level_low,
      createdSec: isoToSec(fvg.created_at),
      mitigatedSec: fvg.mitigated_at ? isoToSec(fvg.mitigated_at) : null,
      tested: fvg.status !== 'active',
      label: 'Fair Value Gap',
    });
  }
  // Drop zones with an unparseable formation time — can't be anchored.
  return out.filter((z) => Number.isFinite(z.createdSec));
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
