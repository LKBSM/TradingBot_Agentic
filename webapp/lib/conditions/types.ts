/**
 * Conditions Scanner — shared types (SC-1).
 *
 * The scanner is a DESCRIPTIVE, present-tense tool: the user composes structural
 * conditions (their "reading") and the app shows on which market/timeframe those
 * conditions are PRESENT right now. No prediction, no outcome, no ranking, no
 * sort by how many conditions a combo meets.
 *
 * These types mirror the backend contract (src/api/routes/conditions_scan.py).
 * The palette itself (families + per-condition controls) is fetched from
 * GET /api/conditions-scan/palette so the interface derives from the schema.
 */

/** The four families conditions are grouped under (fixed order). */
export type Family = 'structure' | 'zones' | 'liquidity' | 'context';

/** Closed set of present-tense condition types. No predictive type exists. */
export type ConditionType =
  // structure
  | 'trend_is'
  | 'higher_tf_agrees'
  | 'last_event_is'
  | 'last_event_age'
  | 'bos_recent_confirmed'
  | 'choch_recent_confirmed'
  // zones
  | 'price_in_ob'
  | 'price_in_fvg'
  | 'price_in_tested_zone'
  | 'zone_untested'
  | 'zone_formed_recent'
  | 'price_near_ob'
  | 'price_near_fvg'
  // liquidity
  | 'price_near_liquidity'
  | 'liquidity_swept_recent'
  | 'liquidity_broken_recent'
  | 'equal_levels_present'
  // context
  | 'market_phase_is'
  | 'volatility_is'
  | 'price_in_range_third'
  | 'session_is';

export type DirectionFilter = 'any' | 'bullish' | 'bearish';
export type LiquiditySideFilter = 'any' | 'bsl' | 'ssl';
export type ZoneKindFilter = 'any' | 'ob' | 'fvg';
export type ScanLogic = 'AND' | 'OR';

// TR-1: structural trend vocabulary (matches backend TrendChoice / TrendValue).
export type TrendChoice = 'bullish' | 'bearish' | 'indeterminate';
// ``distribution`` is intentionally absent — the engine cannot emit it.
export type PhaseChoice = 'accumulation' | 'trend' | 'ranging' | 'expansion';
export type VolatilityChoice = 'low' | 'normal' | 'elevated';
export type EventChoice = 'bos_up' | 'bos_down' | 'choch_up' | 'choch_down';
export type AgeBucketChoice = 'lt10' | '10to50' | 'gt50';
export type RangeThirdChoice = 'bottom' | 'middle' | 'top';
export type EqKindChoice = 'any' | 'highs' | 'lows';
export type SessionChoice = 'asia' | 'london' | 'new_york' | 'overlap';
// Only directional relations are selectable; an indeterminate unit (current or
// higher) makes the comparison non-evaluable, not a target one can require (C1-b).
export type RelationChoice = 'same' | 'opposite';

/** The name of a segmented control — also the ScanCondition field it drives. */
export type ControlName =
  | 'direction'
  | 'max_bars'
  | 'trend'
  | 'phase'
  | 'volatility'
  | 'proximity_pct'
  | 'side'
  | 'zone_kind'
  | 'event'
  | 'age_bucket'
  | 'third'
  | 'eq_kind'
  | 'session'
  | 'relation';

/** One condition in the user's set (wire-shaped: matches the POST body). */
export interface ScanCondition {
  type: ConditionType;
  direction?: DirectionFilter;
  /** Recency window in bars (bos/choch_recent, zone_formed, liquidity_*). */
  max_bars?: number;
  trend?: TrendChoice;
  phase?: PhaseChoice;
  volatility?: VolatilityChoice;
  /** Proximity threshold in % of price (price_near_ob / _fvg / _liquidity). */
  proximity_pct?: number;
  side?: LiquiditySideFilter;
  zone_kind?: ZoneKindFilter;
  event?: EventChoice;
  age_bucket?: AgeBucketChoice;
  third?: RangeThirdChoice;
  eq_kind?: EqKindChoice;
  session?: SessionChoice;
  relation?: RelationChoice;
}

/** The user's saved reading (their conditions + AND/OR logic). */
export interface ConditionsConfig {
  logic: ScanLogic;
  conditions: ScanCondition[];
}

/** A segmented control descriptor (from the schema). */
export interface ControlDescriptor {
  name: ControlName;
  /** All options shown at once as segmented buttons (never a dropdown). */
  values: Array<string | number>;
  default: string | number;
}

/** A palette entry — what the builder may offer. Always present-tense. */
export interface PaletteEntry {
  type: ConditionType;
  family: Family;
  label: string;
  description: string;
  controls: ControlDescriptor[];
  tense: 'present';
  supports_direction?: boolean;
}

/** A condition prepared but NOT offerable yet (blocked), surfaced for honesty. */
export interface BlockedEntry {
  type: string;
  family: Family;
  label: string;
  blocked_reason: string;
  tense: 'present';
}

/** GET /api/conditions-scan/palette response. */
export interface PaletteResponse {
  families: Family[];
  palette: PaletteEntry[];
  blocked: BlockedEntry[];
}

// ── Scan response (matches the backend ConditionsScanResponse) ───────────────

export interface ConditionOutcome {
  type: string;
  label: string;
  met: boolean;
  /**
   * False when the data needed to judge this condition is missing — the
   * NON-EVALUABLE third state, counted neither as met nor as unmet.
   */
  available?: boolean;
  detail: string;
}

export interface ContextZone {
  direction?: string | null;
  level?: number | null;
  validation_status?: string | null;
}

export interface ComboContext {
  trend: string | null;
  market_phase: string | null;
  volatility_observed: string | null;
  mtf_confluence: Record<string, string>;
  mtf_trends?: Record<string, string | null>;
  bos: ContextZone | null;
  choch: ContextZone | null;
  active_order_blocks: number;
  active_fair_value_gaps: number;
  structural_range?: { low: number; high: number } | null;
  news_upcoming: Array<{
    event: string | null;
    impact: string | null;
    time_to_event_min: number | null;
  }>;
}

export interface ComboMatch {
  instrument: string;
  timeframe: string;
  candle_close_ts: string | null;
  close_price: number | null;
  /** True when the combo satisfies the AND/OR logic in full (evaluable only). */
  matched: boolean;
  met_count: number;
  /** Denominator: counts only EVALUABLE conditions. */
  total: number;
  /** How many conditions could not be judged (adjust the denominator). */
  non_evaluable_count?: number;
  conditions_met: ConditionOutcome[];
  conditions_unmet: ConditionOutcome[];
  conditions_non_evaluable?: ConditionOutcome[];
  /** Factual against-signals surfaced even on a full match (« à l'encontre »). */
  context_against?: Array<{ label: string; detail: string }>;
  context: ComboContext;
  bars_behind?: number;
  freshness?: Freshness;
}

/** Reading-freshness tier (see ComboMatch.freshness). */
export type Freshness = 'fresh' | 'aging' | 'stale';

export interface UnavailableCombo {
  instrument: string;
  timeframe: string;
  reason: string;
}

export interface ConditionsScanResponse {
  as_of: string;
  logic: string;
  scanned: number;
  matches: ComboMatch[];
  unavailable: UnavailableCombo[];
}
