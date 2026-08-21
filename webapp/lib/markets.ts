// MKT-1 — the frontend market registry. Everything about a market derives from
// MARKET_SPECS (generated from config/markets.json, the SAME source the backend
// reads). No component enumerates markets: they call these helpers. Adding an
// 81st market = edit the JSON + regenerate, nothing here.

import { MARKET_SPECS, type MarketSpec, type MarketType } from './markets.generated';

export type { MarketSpec, MarketType };
export { MARKET_SPECS };

const BY_ID: Record<string, MarketSpec> = Object.fromEntries(
  MARKET_SPECS.map((s) => [s.id, s]),
);

export function marketSpec(id: string): MarketSpec | null {
  return BY_ID[(id ?? '').toUpperCase()] ?? null;
}

/** All market ids in display order — the supported perimeter. */
export const ALL_MARKET_IDS: readonly string[] = MARKET_SPECS.map((s) => s.id);

/** FR baseline label. Locale-aware components use the `markets.<id>` message. */
export function marketLabel(id: string): string {
  return marketSpec(id)?.label ?? id;
}

/** Conventional display precision for a price on this market (default 2). */
export function marketPriceDecimals(id: string): number {
  return marketSpec(id)?.priceDecimals ?? 2;
}

/** Short mono badge (1-2 chars) for the compact market row. */
export function marketGlyph(id: string): string {
  return marketSpec(id)?.glyph ?? (id ?? '').slice(0, 2);
}

/** Asset class of the market (metal | fx | crypto | index). */
export function marketType(id: string): MarketType | null {
  return marketSpec(id)?.type ?? null;
}

/** Perimeter timeframe ids this market is served on (M1 gate applied by caller). */
export function marketTimeframes(id: string): readonly string[] {
  return marketSpec(id)?.timeframes ?? [];
}

/** id → FR baseline label. Replaces formatters' INSTRUMENT_LABEL. */
export const MARKET_LABEL: Record<string, string> = Object.fromEntries(
  MARKET_SPECS.map((s) => [s.id, s.label]),
);
/** id → price decimals. Replaces the scattered PRICE_DECIMALS copies. */
export const MARKET_PRICE_DECIMALS: Record<string, number> = Object.fromEntries(
  MARKET_SPECS.map((s) => [s.id, s.priceDecimals]),
);
