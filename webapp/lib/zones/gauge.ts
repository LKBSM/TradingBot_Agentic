import type { ZoneLifecycle } from './lifecycle';

/**
 * VZ-3 — the proximity gauge's GEOMETRY, kept pure so the states are
 * unit-testable without a DOM. Presentation (colours, the labels, the écart
 * number itself) lives in the component; this file only decides WHERE things
 * sit on the track and WHICH state the price is in.
 *
 * The window is FIXED around the zone: a margin of half the zone's height on
 * each side, so the band always occupies the middle 50 % of the track
 * (25 %–75 %) regardless of the zone's size. Because the window no longer
 * stretches to swallow a far-away price (the pre-VZ-3 behaviour, which let the
 * band shrink to an invisible sliver), a price beyond the margin is
 * « out of window » — pinned to the edge and marked as such, never faked to
 * scale. The two extent numbers the old gauge printed at its ends are gone:
 * only the zone's own edges and the price are chiffrés.
 */

export type ZoneGaugeState =
  | 'inside' // the price sits within the band
  | 'above' // the price is above the zone, still within the window
  | 'below' // the price is below the zone, still within the window
  | 'outAbove' // the price is above the zone, beyond the window
  | 'outBelow'; // the price is below the zone, beyond the window

export interface ZoneGaugeLayout {
  state: ZoneGaugeState;
  /** Band edges as left-anchored percentages of the track (bandLowPct < bandHighPct). */
  bandLowPct: number;
  bandHighPct: number;
  /** Price marker position along the track, clamped to [0, 100]. */
  pricePct: number;
  /**
   * The measurement bracket, in track percentages, from the reference edge to
   * the price marker — null when the price is inside (no distance) or out of
   * window (a pinned marker is not to scale, so a bracket would lie).
   */
  bracket: { fromPct: number; toPct: number } | null;
  /** The zone edge the distance is measured to ('high' when the price is above the zone). */
  refEdge: 'low' | 'high';
  /** True when the price is beyond the window (pinned to an edge). */
  outOfWindow: boolean;
}

/** Margin added on EACH side of the zone; 0.5 → band fills the middle 50 % of the track. */
export const GAUGE_MARGIN_RATIO = 0.5;

/**
 * Lay out the gauge for a zone and a (finite, positive) price. Callers gate on
 * `zoneProximity(...) !== null` first, so `price` is always a real quote here.
 */
export function zoneGaugeLayout(zone: ZoneLifecycle, price: number): ZoneGaugeLayout {
  const low = zone.levelLow;
  const high = zone.levelHigh;
  const height = Math.max(high - low, Number.EPSILON);
  const margin = height * GAUGE_MARGIN_RATIO;
  const eLo = low - margin;
  const eHi = high + margin;
  const span = eHi - eLo || 1; // = 2 × height
  const pct = (v: number) => ((v - eLo) / span) * 100;

  const bandLowPct = pct(low);
  const bandHighPct = pct(high);

  // Inside the band — no distance, no bracket.
  if (price >= low && price <= high) {
    return {
      state: 'inside',
      bandLowPct,
      bandHighPct,
      pricePct: pct(price),
      bracket: null,
      refEdge: 'high',
      outOfWindow: false,
    };
  }

  // Above the zone (marker to the right, measured at the upper edge).
  if (price > high) {
    if (price > eHi) {
      return {
        state: 'outAbove',
        bandLowPct,
        bandHighPct,
        pricePct: 100,
        bracket: null,
        refEdge: 'high',
        outOfWindow: true,
      };
    }
    const pricePct = pct(price);
    return {
      state: 'above',
      bandLowPct,
      bandHighPct,
      pricePct,
      bracket: { fromPct: bandHighPct, toPct: pricePct },
      refEdge: 'high',
      outOfWindow: false,
    };
  }

  // Below the zone (marker to the left, measured at the lower edge).
  if (price < eLo) {
    return {
      state: 'outBelow',
      bandLowPct,
      bandHighPct,
      pricePct: 0,
      bracket: null,
      refEdge: 'low',
      outOfWindow: true,
    };
  }
  const pricePct = pct(price);
  return {
    state: 'below',
    bandLowPct,
    bandHighPct,
    pricePct,
    bracket: { fromPct: pricePct, toPct: bandLowPct },
    refEdge: 'low',
    outOfWindow: false,
  };
}
