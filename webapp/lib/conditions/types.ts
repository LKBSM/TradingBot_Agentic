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
  | 'price_in_ob'
  | 'price_in_fvg'
  | 'ob_fvg_confluence'
  | 'bos_recent_confirmed';

export type DirectionFilter = 'any' | 'bullish' | 'bearish';
export type ScanLogic = 'AND' | 'OR';

/** One condition in the user's set (wire-shaped: matches the POST body). */
export interface ScanCondition {
  type: ConditionType;
  direction: DirectionFilter;
  /** Recency window in bars for `bos_recent_confirmed` (ignored by others). */
  max_bars?: number;
}

/** The user's saved configuration (their conditions + AND/OR logic). */
export interface ConditionsConfig {
  logic: ScanLogic;
  conditions: ScanCondition[];
}

/** A palette entry — what the builder may offer. Always present-tense. */
export interface PaletteEntry {
  type: ConditionType;
  label: string;
  description: string;
  supportsDirection: boolean;
  tense: 'present';
}

// ── Scan response (matches the backend ConditionsScanResponse) ───────────────

export interface ConditionOutcome {
  type: string;
  label: string;
  met: boolean;
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
