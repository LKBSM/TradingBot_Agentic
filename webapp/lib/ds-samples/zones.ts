/**
 * DS-1 — Zone lifecycle samples for the design gallery.
 *
 * Derived from the REAL XAUUSD H4 reading (readings.ts) with the SAME pure
 * function the /zones surface uses (`collectZones`) — no reimplementation, no
 * network. The zones therefore carry real levels, statuses and timestamps.
 *
 * `referencePrice` is the reading's real close (4 043,24), so the proximity gauge
 * shows real distances — including the edge case where the price sits INSIDE a
 * band (`SAMPLE_ZONE_PRICE_INSIDE`), which is what stresses the gauge.
 */
import { collectZones, collectConsumedZones } from '@/lib/zones/lifecycle';
import type { ZoneLifecycle } from '@/lib/zones/lifecycle';
import { SAMPLE_READING_XAU_H4 } from './readings';

export const SAMPLE_ZONES_XAU_H4: ZoneLifecycle[] = collectZones(SAMPLE_READING_XAU_H4.structure);
export const SAMPLE_CONSUMED_ZONES_XAU_H4: ZoneLifecycle[] = collectConsumedZones(
  SAMPLE_READING_XAU_H4.structure,
);
export const SAMPLE_LIQUIDITY_XAU_H4 = SAMPLE_READING_XAU_H4.structure.liquidity_pools ?? [];
export const SAMPLE_ZONE_REFERENCE_PRICE = SAMPLE_READING_XAU_H4.header.close_price;

const price = SAMPLE_ZONE_REFERENCE_PRICE;

/** A zone the price sits INSIDE — the edge case that stresses the proximity gauge. */
export const SAMPLE_ZONE_PRICE_INSIDE: ZoneLifecycle | undefined = SAMPLE_ZONES_XAU_H4.find(
  (z) => z.levelLow <= price && price <= z.levelHigh,
);
/** A zone entirely ABOVE the price. */
export const SAMPLE_ZONE_ABOVE: ZoneLifecycle | undefined = SAMPLE_ZONES_XAU_H4.find(
  (z) => z.levelLow > price,
);
/** A zone entirely BELOW the price. */
export const SAMPLE_ZONE_BELOW: ZoneLifecycle | undefined = SAMPLE_ZONES_XAU_H4.find(
  (z) => z.levelHigh < price,
);
/** A zone that has been touched / tested at least once. */
export const SAMPLE_ZONE_TESTED: ZoneLifecycle | undefined = SAMPLE_ZONES_XAU_H4.find(
  (z) => z.tested,
);
/** A zone still untouched. */
export const SAMPLE_ZONE_UNTOUCHED: ZoneLifecycle | undefined = SAMPLE_ZONES_XAU_H4.find(
  (z) => !z.tested,
);
