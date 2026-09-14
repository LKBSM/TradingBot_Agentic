/**
 * LP-1 / LP-2 — SINGLE SOURCE OF TRUTH for the perimeter figures the home page
 * advertises.
 *
 * LP-3 removed the four-tile stats banner from the hero (founder decision), but
 * these figures did NOT become decorative: they are what the pricing card, the
 * FAQ and the "how it works" copy claim about the real perimeter. They live
 * here, tied to the real config they mirror, so the copy can never drift into
 * fiction. A test asserts they still match their sources.
 *
 * Reality check:
 *   · markets      = the market registry              (config/markets.json)
 *   · timeframes   = what the selector actually SHOWS  (DISPLAY_TIMEFRAMES)
 *   · combinations = markets × timeframes              (mirrors enabled_combos())
 *   · conditions   = scanner palette length = 22       (conditions palette)
 *   · structures   = distinct structure families the detection surfaces = 7
 *
 * The maquette claimed "80 marchés / 480 combinaisons / 21 conditions". Three of
 * those four were fiction; these are the true figures.
 *
 * LP-3 — the first two are now DERIVED rather than written down, because they
 * had silently drifted. Decision DATA-1 (2026-08-16) cut M1 from the perimeter:
 * the selector shows five units and the scanner sweeps 2 × 5 = 10 combinations,
 * while this file still said 6 and 12 and the copy repeated it in nine
 * languages. A number the landing advertises must come from the thing it
 * describes; if M1 is ever re-enabled, the page follows on its own.
 */

/**
 * LP-2 (§C4/§D) — the SEVEN distinct structure families the product detects and
 * renders. This is the single source behind the banner's "structures détectées"
 * tile, so the figure can never be a bare literal. Each id maps to a real,
 * user-visible layer/family across /app, /zones and the scanner palette:
 *   order_block, fair_value_gap  → zones (OB / FVG)
 *   bos, choch                   → structure breaks (BOS / CHOCH)
 *   bsl_pocket, ssl_pocket       → liquidity pockets (buy-/sell-side)
 *   equal_levels                 → equal highs / lows (EQH / EQL)
 * A test asserts LANDING_STATS.structures === STRUCTURE_TYPES.length.
 */
import { ALL_MARKET_IDS } from '@/lib/markets';
import { DISPLAY_TIMEFRAMES } from '@/lib/market-reading/perimeter';

export const STRUCTURE_TYPES = [
  'order_block',
  'fair_value_gap',
  'bos',
  'choch',
  'bsl_pocket',
  'ssl_pocket',
  'equal_levels',
] as const;

export type StructureType = (typeof STRUCTURE_TYPES)[number];

export const LANDING_STATS = {
  markets: ALL_MARKET_IDS.length,
  timeframes: DISPLAY_TIMEFRAMES.length,
  combinations: ALL_MARKET_IDS.length * DISPLAY_TIMEFRAMES.length,
  conditions: 22,
  structures: STRUCTURE_TYPES.length,
} as const;

export type LandingStatKey = keyof typeof LANDING_STATS;
