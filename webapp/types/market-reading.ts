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

/**
 * Structural market trend on the reading timeframe (TR-1). Derived from the
 * engine's last non-contradicted BOS/CHOCH — a direction, or `indeterminate`
 * when NO structural break exists in the analysed history. `neutral`/`ranging`
 * are gone (consolidation now lives on the Phase tile only).
 */
export type TrendValue = 'bullish' | 'bearish' | 'indeterminate';

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
export type MTFBiasValue = 'bullish' | 'bearish' | 'indeterminate';

/** Kind of structural break that anchors a trend (TR-1). */
export type TrendReferenceKind = 'bos' | 'choch';

/**
 * The structural event anchoring the current `trend` — the last CHOCH (change of
 * character) that set the direction, or the last BOS when no CHOCH exists. Lets
 * the Trend tile name WHY it reads as it does (« depuis le CHOCH haussier du
 * 24 juil. »), telling the same story as the Maturité tile. Absent (null) when
 * the trend is `indeterminate`.
 */
export interface TrendReference {
  kind: TrendReferenceKind;
  direction: 'bullish' | 'bearish';
  level: number;
  broken_at: string;
  bars_ago?: number | null;
}

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

/** External liquidity side: buy-side (above) / sell-side (below). */
export type LiquiditySide = 'bsl' | 'ssl';

/** Liquidity pocket geometry. */
export type LiquidityKind =
  | 'equal_highs'
  | 'equal_lows'
  | 'range_high'
  | 'range_low';

/** Liquidity pocket lifecycle (descriptive, factual). */
export type LiquidityStatus = 'intact' | 'swept' | 'broken';

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
  /**
   * Number of bars actually analysed for this reading — the per-timeframe live
   * window (MT-D1). Optional: older payloads omit it, and the UI then falls back
   * to the legacy `TREND_WINDOW_BARS` (500) constant.
   */
  analysis_window_bars?: number | null;
}

// ─── Structure (Smart Money Concepts) ────────────────────────────────────────

export interface BOSRecent {
  direction: Direction;
  level: number;
  broken_at: string;
  validation_status: ValidationStatus;
  /** Whole ANALYSED bars between the break and the last bar of the window (DG-1).
   *  Absent on older payloads → callers fall back to a wall-clock estimate. */
  bars_ago?: number | null;
}

export interface CHOCHRecent {
  direction: Direction;
  level: number;
  broken_at: string;
  validation_status: ValidationStatus;
  bars_ago?: number | null;
}

/**
 * VZ-1 — outcome of ONE observed price contact with a zone. Strictly factual,
 * NEVER a judgement:
 *   · `edge_touch` — price reached the near edge without really penetrating.
 *   · `entry_exit` — price entered the band then left through the SAME edge.
 *   · `traversal`  — price crossed the band / fully filled the gap (consumed it).
 *   · `inside`     — price is CURRENTLY within the band (ongoing).
 */
export type ContactOutcome = 'edge_touch' | 'entry_exit' | 'traversal' | 'inside';

export interface ZoneContact {
  /** Entry timestamp of this contact. */
  at: string;
  /** Deepest price reached into the band on this contact ("niveau atteint"). */
  level: number;
  outcome: ContactOutcome;
}

/** VZ-1 — the structural break an order block precedes (what makes it an OB). */
export interface ZoneOrigin {
  kind: 'bos' | 'choch';
  direction: Direction;
  at: string;
  level: number;
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
  /** VZ-1 — per-contact ledger (absent on older payloads → treat as empty). */
  contacts?: ZoneContact[];
  /** VZ-1 — the BOS/CHOCH break this OB precedes (absent → not associated). */
  origin?: ZoneOrigin | null;
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
  /** VZ-1 — per-contact ledger (absent on older payloads → treat as empty). */
  contacts?: ZoneContact[];
  user_flagged: boolean;
}

export interface RetestInProgress {
  level: number;
  type: RetestType;
  started_at: string;
}

/**
 * External liquidity pocket — equal highs/lows or a range extreme. Strictly
 * DESCRIPTIVE (niveau 1.5): WHERE resting liquidity sits and WHETHER that level
 * has been intact / swept / broken. No target, draw, bias or probability. Mirror
 * of the Pydantic `LiquidityPool`.
 */
export interface LiquidityPool {
  id: string;
  /** `bsl` = buy-side (above), `ssl` = sell-side (below). */
  side: LiquiditySide;
  kind: LiquidityKind;
  /** Resting-liquidity price level. */
  level: number;
  /** Swing points forming the pocket (1 for a range extreme). */
  touches: number;
  /** True when at/beyond the current range's extreme swing. */
  is_external: boolean;
  status: LiquidityStatus;
  created_at: string;
  /** First bar that wicked through and closed back inside. null unless swept. */
  swept_at?: string | null;
  /** First bar that closed net through the level. null unless broken (terminal). */
  broken_at?: string | null;
  user_flagged: boolean;
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
  /**
   * VZ-1 — a bounded set of the most recently CONSUMED zones (invalidated OB /
   * filled FVG) the live lists drop once consumed, for the /zones « Comblées »
   * group. Absent on older payloads (treat as empty). The /app surface ignores
   * these; each carries its full contact ledger ending with a `traversal`.
   */
  consumed_order_blocks?: OrderBlock[];
  consumed_fair_value_gaps?: FairValueGap[];
  /**
   * External liquidity pockets (equal highs/lows + range extremes) with
   * intact/swept/broken state. Read-only/descriptive twin of order_blocks /
   * fair_value_gaps. Absent on older payloads (treat as empty).
   */
  liquidity_pools?: LiquidityPool[];
  retest_in_progress?: RetestInProgress | null;
}

// ─── Regime ────────────────────────────────────────────────────────────────

/**
 * Numeric intermediates behind `volatility_observed`, so the proof panel can let
 * a reader redo the operation. `ratio = recent_avg / baseline_avg`; the category
 * is `low` below `threshold_low`, `elevated` above `threshold_high`, else
 * `normal`. Averages are mean True Ranges (high − low) over the last `recent_n`
 * candles and the `baseline_n` candles before them (all remaining, not a fixed
 * 20). Absent when the window is too short (< 14 candles) or on older payloads.
 */
export interface VolatilityDetail {
  recent_avg: number;
  baseline_avg: number;
  ratio: number;
  recent_n: number;
  baseline_n: number;
  threshold_low: number;
  threshold_high: number;
}

export interface MarketReadingRegime {
  trend: TrendValue;
  volatility_observed: VolatilityObserved;
  market_phase: MarketPhase;
  /** Multi-timeframe directional biases, keyed by `MTFTimeframeKey`. */
  mtf_confluence: Partial<Record<MTFTimeframeKey, MTFBiasValue>>;
  /** Numeric proof behind `volatility_observed`; absent on short windows. */
  volatility_detail?: VolatilityDetail | null;
  /** Structural event anchoring `trend` (TR-1); null when `indeterminate`. */
  trend_reference?: TrendReference | null;
}

// ─── Events ────────────────────────────────────────────────────────────────

export interface NewsUpcoming {
  event: string;
  scheduled_at: string;
  time_to_event_min: number;
  impact: ImpactLevel;
  currency: string;
  potential_effect_description: string;
  /** Deterministic release id — deep-links the App news row to /actualites. */
  event_id?: string | null;
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
  event_id?: string | null;
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
  /**
   * MC-1 server-computed market status. Present on live readings, absent on
   * static/landing fixtures (the UI then falls back to the client heuristic).
   * The single source of truth for the App badge, Scanner and agent — never the
   * client clock.
   */
  market_status?: MarketStatusPayload | null;
  /**
   * RG-1c calendar reference levels — day/week open + previous day/week extremes,
   * aggregated SERVER-SIDE over the MC-1 trading calendar (one definition of « a
   * trading day », per instrument, in NY wall-clock). Each value is null when its
   * period is not fully covered by the cached candles; `day_complete` /
   * `week_complete` flag whether the previous-day / previous-week window was
   * fully covered so the panel can name « données insuffisantes ». Absent on
   * static fixtures.
   */
  reference_levels?: ReferenceLevelsPayload | null;
}

/** Server-aggregated calendar reference levels (reference_levels.to_dict). */
export interface ReferenceLevelsPayload {
  day_open: number | null;
  week_open: number | null;
  prev_day_high: number | null;
  prev_day_low: number | null;
  prev_week_high: number | null;
  prev_week_low: number | null;
  day_complete: boolean;
  week_complete: boolean;
}

/** Server market status (src/intelligence/market_calendar.MarketStatus.to_dict). */
export type MarketState =
  | 'open'
  | 'closed_weekend'
  | 'closed_holiday'
  | 'daily_break'
  | 'data_lagged';

export interface MarketStatusPayload {
  state: MarketState;
  reason: string;
  instrument: string;
  timeframe: string;
  /** ISO-8601 UTC of the last fully-closed candle, or null. */
  last_close_ts: string | null;
  /** ISO-8601 UTC of the next open, or null (open / 24-7 market). */
  next_open_ts: string | null;
  bars_behind: number | null;
  /** 24/7 market (crypto) — no session découpage, no weekly close. */
  continuous?: boolean;
  /** IANA zone the session windows are expressed in (e.g. America/New_York). */
  session_tz?: string;
  /** Named intraday session windows (« HH:MM » ET). Empty ⇒ continuous. */
  sessions?: { name: string; start: string; end: string }[];
  /** Weekly close, Python weekday (Mon=0…Sun=6) + « HH:MM » ET. */
  weekly_close?: { weekday: number; time: string } | null;
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
