/**
 * LP-1 — SINGLE SOURCE OF TRUTH for the home-page stats banner.
 *
 * Every number the landing advertises about the product's real perimeter lives
 * here, tied to the real config it mirrors, so the banner can never drift into
 * fiction. A test (home.stats.test.ts) asserts the banner renders THESE values
 * and nothing hard-coded elsewhere.
 *
 * Reality check (verified 2026-08-01, do not inflate):
 *   · markets      = SUPPORTED_INSTRUMENTS  → XAUUSD, EURUSD          (perimeter.ts)
 *   · timeframes   = timeframes with perimeter:true → M1..D1 = 6      (config/timeframes.json)
 *   · combinations = markets × timeframes = 2 × 6 = 12                (enabled_combos)
 *   · conditions   = scanner palette length = 22                      (conditions palette)
 *
 * The maquette claimed "80 marchés / 480 combinaisons / 21 conditions". Three of
 * those four were fiction; these are the true figures.
 */
export const LANDING_STATS = {
  markets: 2,
  timeframes: 6,
  combinations: 12,
  conditions: 22,
} as const;

export type LandingStatKey = keyof typeof LANDING_STATS;

/** Order the four tiles appear in the banner. */
export const LANDING_STAT_ORDER: readonly LandingStatKey[] = [
  'markets',
  'combinations',
  'conditions',
  'timeframes',
];
