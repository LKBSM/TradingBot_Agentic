/**
 * MarketReading — TypeScript mirror of the Pydantic v2.0.0 contract.
 *
 * Source of truth: `src/intelligence/market_reading_schema.py` (Chantier 2,
 * aligned on `docs/architecture/MIA_MARKETS_V2_VISION.md` §2.3). The literal
 * unions and field names below MUST stay in sync with the backend schema.
 *
 * Posture rule (niveau 1.5 strict): this contract DESCRIBES market conditions
 * factually. It never recommends action — no entry / stop / take-profit / lot
 * fields, and no synthetic 0-100 conviction score (that would be a niveau-2
 * slide). The webapp consumes this object as-is; there is no mapper towards an
 * intermediate format.
 *
 * Timestamps are ISO-8601 strings (the backend serialises `datetime` to JSON).
 */

// ─── Literal vocabularies (mirror the Pydantic Literal aliases) ──────────────

/** SMC event direction. */
export type Direction = 'bullish' | 'bearish';

/** Validation state of a BOS / CHOCH break. */
export type ValidationStatus = 'confirmed' | 'pending' | 'invalidated';

/** News impact level. */
export type ImpactLevel = 'low' | 'medium' | 'high';

/** Direction of a news surprise vs forecast. */
export type SurpriseDirection = 'beat' | 'miss' | 'in_line';

/** Observed market trend on the reading timeframe. */
export type TrendValue = 'bullish' | 'bearish' | 'neutral' | 'ranging';

/** Observed volatility bucket. */
export type VolatilityObserved = 'low' | 'normal' | 'elevated';

/** Descriptive market phase. */
export type MarketPhase =
  | 'accumulation'
  | 'distribution'
  | 'trend'
  | 'ranging'
  | 'expansion';

/** Per-timeframe directional bias used in MTF confluence. */
export type MTFBiasValue = 'bullish' | 'bearish' | 'neutral' | 'ranging';

/** Order Block lifecycle status. */
export type OBStatus = 'active' | 'mitigated' | 'invalidated';

/** Fair Value Gap lifecycle status. */
export type FVGStatus = 'active' | 'partially_filled' | 'filled';

/** Order Block importance bucket. */
export type OBImportance = 'low' | 'medium' | 'high';

/** Retest target type. */
export type RetestType =
  | 'bos_retest'
  | 'choch_retest'
  | 'ob_retest'
  | 'fvg_retest';

/** Provenance of the synthesised conditions description. */
export type DescriptionSource = 'haiku_generated' | 'template_fallback';

/** Valid MTF timeframe keys for the confluence map (`VALID_MTF_KEYS`). */
export type MTFTimeframeKey = 'm15' | 'h1' | 'h4' | 'd1' | 'w1';

// ─── Header ──────────────────────────────────────────────────────────────────

export interface MarketReadingHeader {
  instrument: string;
  timeframe: string;
  /** ISO-8601 timestamp of the candle close this reading describes. */
  candle_close_ts: string;
  close_price: number;
}

// ─── Structure (Smart Money Concepts) ────────────────────────────────────────

export interface BOSRecent {
  direction: Direction;
  level: number;
  broken_at: string;
  validation_status: ValidationStatus;
}

export interface CHOCHRecent {
  direction: Direction;
  level: number;
  broken_at: string;
  validation_status: ValidationStatus;
}

export interface OrderBlock {
  id: string;
  /** Optional — populated by the production SMC scanner, omitted in the doc example. */
  direction?: Direction | null;
  level_high: number;
  level_low: number;
  importance: OBImportance;
  status: OBStatus;
  created_at: string;
  tested: boolean;
  /**
   * Timestamp of first interaction (mitigation point). null/absent while the
   * zone is untouched. Bound the box created_at → mitigated_at; for active
   * zones extend to the current price. Descriptive, never predictive.
   */
  mitigated_at?: string | null;
  user_flagged: boolean;
}

export interface FairValueGap {
  id: string;
  /** Optional — populated by the production SMC scanner, omitted in the doc example. */
  direction?: Direction | null;
  level_high: number;
  level_low: number;
  status: FVGStatus;
  created_at: string;
  tested: boolean;
  /** First-entry (partial-fill) timestamp; same box-bounding role as OrderBlock.mitigated_at. */
  mitigated_at?: string | null;
  /**
   * Price the gap has been penetrated to — the deepest wick into the band so
   * far (within [level_low, level_high]). null/absent while active. Read-only:
   * the chart shrinks a partially-filled box to the still-open portion using
   * this, so the rectangle stops "just under the wicks". Never predictive.
   */
  fill_level?: number | null;
  user_flagged: boolean;
}

export interface RetestInProgress {
  level: number;
  type: RetestType;
  started_at: string;
}

export interface MarketReadingStructure {
  bos?: BOSRecent | null;
  choch?: CHOCHRecent | null;
  /**
   * Discrete BOS / CHOCH break events over the window, most-recent first
   * (read-only, descriptive). The engine detects many breaks but only the
   * last-bar one surfaced via `bos`/`choch`; these lists carry the recent
   * history so the chart can mark each break. Absent on older payloads.
   */
  bos_events?: BOSRecent[];
  choch_events?: CHOCHRecent[];
  order_blocks: OrderBlock[];
  fair_value_gaps: FairValueGap[];
  retest_in_progress?: RetestInProgress | null;
}

// ─── Regime ────────────────────────────────────────────────────────────────

export interface MarketReadingRegime {
  trend: TrendValue;
  volatility_observed: VolatilityObserved;
  market_phase: MarketPhase;
  /** Multi-timeframe directional biases, keyed by `MTFTimeframeKey`. */
  mtf_confluence: Partial<Record<MTFTimeframeKey, MTFBiasValue>>;
}

// ─── Events ────────────────────────────────────────────────────────────────

export interface NewsUpcoming {
  event: string;
  scheduled_at: string;
  time_to_event_min: number;
  impact: ImpactLevel;
  currency: string;
  potential_effect_description: string;
}

export interface NewsJustPublished {
  event: string;
  published_at: string;
  actual?: number | null;
  forecast?: number | null;
  previous?: number | null;
  surprise_direction?: SurpriseDirection | null;
  currency: string;
  impact: ImpactLevel;
  potential_effect_description: string;
}

export interface TechnicalTriggerRecent {
  /** Composite `<event>_<tf>[_<direction>]` code (see TRIGGER_TYPE_PATTERN). */
  type: string;
  occurred_at: string;
  minutes_ago: number;
}

export interface MarketReadingEvents {
  news_upcoming: NewsUpcoming[];
  news_just_published: NewsJustPublished[];
  technical_triggers_recent: TechnicalTriggerRecent[];
}

// ─── Conditions ──────────────────────────────────────────────────────────────

export interface MarketReadingConditions {
  tags: string[];
  /** Plain-language synthesis (≤ 280 chars, DESCRIPTION_MAX_LENGTH). */
  description: string;
  description_source: DescriptionSource;
}

// ─── Root ────────────────────────────────────────────────────────────────────

export interface MarketReading {
  /**
   * Note: the backend places `schema_version` at the ROOT of MarketReading
   * (default "2.0.0"), not inside `header`. The mission brief listed it under
   * the header — we follow the Pydantic source of truth here.
   */
  schema_version: string;
  header: MarketReadingHeader;
  structure: MarketReadingStructure;
  regime: MarketReadingRegime;
  events: MarketReadingEvents;
  conditions: MarketReadingConditions;
}

// ─── Chart feed (GET /api/candles) ────────────────────────────────────────────

/**
 * One OHLC candle as served by GET /api/candles. `time` is a UTC epoch in
 * SECONDS (lightweight-charts' UTCTimestamp). Strictly descriptive — the series
 * stops at the last fully-closed candle, never a forward projection.
 */
export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  /** Tick/real volume when available (0 when the provider omits it). */
  volume?: number;
}

/** Envelope returned by GET /api/candles. */
export interface CandlesResponse {
  instrument: string;
  timeframe: string;
  candles: Candle[];
}

// ─── Convenience helpers ──────────────────────────────────────────────────────

export function isBullishTrend(r: Pick<MarketReadingRegime, 'trend'>): boolean {
  return r.trend === 'bullish';
}

export function isBearishTrend(r: Pick<MarketReadingRegime, 'trend'>): boolean {
  return r.trend === 'bearish';
}
