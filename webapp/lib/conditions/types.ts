/**
 * Conditions Scanner — shared types.
 *
 * The scanner is a DESCRIPTIVE, present-tense tool: the user composes structural
 * conditions (their "strategy") and the app shows on which market/timeframe those
 * conditions are PRESENT right now. No prediction, no outcome, no ranking.
 *
 * These types mirror the backend contract (src/api/routes/conditions_scan.py).
 */

/** Closed set of present-tense condition types. No predictive type exists. */
export type ConditionType =
  | 'mtf_aligned'
  | 'trend_is'
  | 'market_phase_is'
  | 'volatility_is'
  | 'price_in_ob'
  | 'price_in_fvg'
  | 'ob_fvg_confluence'
  | 'bos_recent_confirmed'
  | 'choch_recent_confirmed'
  | 'retest_in_progress';

export type DirectionFilter = 'any' | 'bullish' | 'bearish';
export type ScanLogic = 'AND' | 'OR';

export type TrendChoice = 'bullish' | 'bearish' | 'ranging' | 'neutral';
export type PhaseChoice =
  | 'accumulation'
  | 'distribution'
  | 'trend'
  | 'ranging'
  | 'expansion';
export type VolatilityChoice = 'low' | 'normal' | 'elevated';

/** One condition in the user's set (wire-shaped: matches the POST body). */
export interface ScanCondition {
  type: ConditionType;
  direction?: DirectionFilter;
  /** Recency window in bars for bos/choch_recent_confirmed (ignored by others). */
  max_bars?: number;
  /** Regime selectors (used by trend_is / market_phase_is / volatility_is). */
  trend?: TrendChoice;
  phase?: PhaseChoice;
  volatility?: VolatilityChoice;
}

/** The user's saved configuration (their conditions + AND/OR logic). */
export interface ConditionsConfig {
  logic: ScanLogic;
  conditions: ScanCondition[];
}

/** Which input controls the builder renders for a condition. */
export type ControlKind = 'direction' | 'bars' | 'trend' | 'phase' | 'volatility';

/** A palette entry — what the builder may offer. Always present-tense. */
export interface PaletteEntry {
  type: ConditionType;
  label: string;
  description: string;
  /** The selectors the builder shows when this condition is picked. */
  controls: ControlKind[];
  tense: 'present';
}

// ── Scan response (matches the backend ConditionsScanResponse) ───────────────

export interface ConditionOutcome {
  type: string;
  label: string;
  met: boolean;
  /**
   * False when the data needed to judge this condition is missing (e.g. a
   * sibling timeframe has no reading yet) — distinct from "judged and not met".
   * Defaults to true for backward compatibility with older payloads.
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
  /** Legacy higher-timeframe bias (incomplete by construction — prefer mtf_trends). */
  mtf_confluence: Record<string, string>;
  /** Authoritative per-timeframe trend (each TF's own regime.trend): h4/h1/m15. */
  mtf_trends?: Record<string, string | null>;
  bos: ContextZone | null;
  choch: ContextZone | null;
  active_order_blocks: number;
  active_fair_value_gaps: number;
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
  /** True when the combo satisfies the AND/OR logic in full. */
  matched: boolean;
  met_count: number;
  total: number;
  conditions_met: ConditionOutcome[];
  conditions_unmet: ConditionOutcome[];
  context: ComboContext;
}

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
