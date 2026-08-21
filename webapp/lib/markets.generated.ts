// AUTO-GENERATED from config/markets.json by scripts/gen_markets.mjs.
// DO NOT EDIT BY HAND. Run `node scripts/gen_markets.mjs` after editing the JSON.
export type MarketType = 'metal' | 'fx' | 'crypto' | 'index';

export interface MarketSpec {
  id: string;
  label: string;
  symbol: string;
  type: MarketType;
  priceDecimals: number;
  glyph: string;
  timeframes: readonly string[];
  index: number;
}

export const MARKET_SPECS: readonly MarketSpec[] = [
  { id: "XAUUSD", label: "Or (XAU/USD)", symbol: "XAUUSD", type: "metal", priceDecimals: 2, glyph: "Au", timeframes: ["M1", "M5", "M15", "H1", "H4", "D1"], index: 0 },
  { id: "EURUSD", label: "Euro / Dollar (EUR/USD)", symbol: "EURUSD", type: "fx", priceDecimals: 5, glyph: "€", timeframes: ["M1", "M5", "M15", "H1", "H4", "D1"], index: 1 },
];
