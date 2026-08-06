/**
 * LP-1 / LP-2 — SINGLE SOURCE OF TRUTH for the home-page stats banner.
 *
 * Every number the landing advertises about the product's real perimeter lives
 * here, tied to the real config it mirrors, so the banner can never drift into
 * fiction. A test (home.test.tsx) asserts the banner renders THESE values and
 * nothing hard-coded elsewhere.
 *
 * Reality check (verified 2026-08-06, do not inflate):
 *   · markets      = SUPPORTED_INSTRUMENTS  → XAUUSD, EURUSD          (perimeter.ts)
 *   · timeframes   = timeframes with perimeter:true → M1..D1 = 6      (config/timeframes.json)
 *   · combinations = markets × timeframes = 2 × 6 = 12                (enabled_combos)
 *   · conditions   = scanner palette length = 22                      (conditions palette)
 *   · structures   = distinct structure families the detection surfaces = 7
 *
 * The maquette claimed "80 marchés / 480 combinaisons / 21 conditions". Three of
 * those four were fiction; these are the true figures.
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
  markets: 2,
  timeframes: 6,
  combinations: 12,
  conditions: 22,
  structures: STRUCTURE_TYPES.length,
} as const;

export type LandingStatKey = keyof typeof LANDING_STATS;

/**
 * Order the four tiles appear in the banner (LP-2 v3):
 * markets · timeframes · conditions · structures.
 * `combinations` stays in LANDING_STATS (a real, tested figure reused elsewhere)
 * but is not one of the four banner tiles.
 */
export const LANDING_STAT_ORDER: readonly LandingStatKey[] = [
  'markets',
  'timeframes',
  'conditions',
  'structures',
];
