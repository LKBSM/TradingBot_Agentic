/**
 * InsightSignalV2 — TypeScript mirror of the Pydantic v2.1.0 contract.
 *
 * Source of truth: `src/api/insight_signal_v2.py` + the field-level reference
 * `docs/value/client_information_explained.txt` (2026-05-17). The literal
 * unions and field names below MUST stay in sync with the backend schema.
 *
 * Numeric units follow the backend convention (pips, percentages 0-100,
 * ISO-8601 strings for timestamps, ratios in [0, 1] unless noted).
 *
 * Posture rule: this contract DESCRIBES the market state. It never carries
 * entry / stop / take-profit / lot-size fields (UE 2024/2811 compliance).
 */

// ─── A. Identity ────────────────────────────────────────────────────────────
export type Instrument =
  | 'XAUUSD'
  | 'EURUSD'
  | 'BTCUSD'
  | 'US500'
  | 'GBPUSD'
  | 'USDJPY';

export type Timeframe = 'M1' | 'M5' | 'M15' | 'M30' | 'H1' | 'H4' | 'D1' | 'W1';

// ─── B. Direction ───────────────────────────────────────────────────────────
export type Direction = 'BULLISH_SETUP' | 'BEARISH_SETUP' | 'NEUTRAL';

// ─── C. Conviction ──────────────────────────────────────────────────────────
export type ConvictionLabel =
  | 'weak'
  | 'moderate'
  | 'strong'
  | 'institutional';

// ─── D. Uncertainty ─────────────────────────────────────────────────────────
export interface UncertaintyContext {
  /** Lower bound of the conformal interval on the 0-100 conviction scale. */
  conformal_lower: number;
  /** Upper bound of the conformal interval on the 0-100 conviction scale. */
  conformal_upper: number;
  /** Nominal miscoverage rate (default 0.10 → 90% interval). */
  coverage_alpha: number;
  /** Rolling calibration buffer size. */
  n_calibration: number;
  /** Empirical coverage rate observed on past intervals (monitoring). */
  empirical_coverage: number;
}

// ─── E. Structure readout (SMC) ─────────────────────────────────────────────
export type RetestState = 'idle' | 'awaiting' | 'armed' | 'consumed';

export interface StructureReadout {
  /** Price of the swing high/low broken by BOS. */
  bos_level: number | null;
  /** Bars elapsed since the BOS event. 0 = fresh break. */
  bos_event_age_bars: number | null;
  /** True if a Change-of-Character (CHOCH) preceded the BOS. */
  choch_present: boolean;
  /** Fair Value Gap zone [low, high], if any. */
  fvg_zone: [number, number] | null;
  /** FVG size normalized by ATR(14). */
  fvg_size_atr: number | null;
  /** Order Block zone [low, high], if any. */
  ob_zone: [number, number] | null;
  /** OB strength normalized in [0, 1]. */
  ob_strength: number | null;
  /** Retest state machine. */
  retest_state: RetestState;
  /** Price level beyond which the SMC thesis is invalidated. */
  structural_invalidation: number | null;
  /** Aspirational fields — null in current backend. */
  liquidity_zone_upper: [number, number] | null;
  liquidity_zone_lower: [number, number] | null;
}

// ─── F. Regime readout ──────────────────────────────────────────────────────
export type HmmLabel =
  | 'trend_bullish'
  | 'trend_bearish'
  | 'range_low_vol'
  | 'high_vol_stress';

export type RegimeGateDecision = 'TRADE' | 'REDUCE' | 'BLOCK';

export interface RegimeReadout {
  hmm_label: HmmLabel;
  /** P(state | observations) — posterior of the chosen hidden state. */
  hmm_posterior: number;
  /** BOCPD change-point probability for the current bar. */
  bocpd_changepoint_prob: number;
  /** Posterior-weighted expected run-length until next regime shift. */
  expected_run_length: number;
  /** Jump component share of realized variance (Barndorff-Nielsen). */
  jump_ratio: number;
  regime_gate_decision: RegimeGateDecision;
}

// ─── G. Volatility readout ──────────────────────────────────────────────────
export type VolatilityRegime = 'low' | 'normal' | 'high';

export interface VolatilityReadout {
  regime: VolatilityRegime;
  /** HAR-RV blended forecast in pips. */
  forecast_atr_pips: number;
  /** Wilder ATR(14) baseline in pips. */
  naive_atr_pips: number;
  /** 100 × (forecast - naive) / naive — positive = expansion. */
  forecast_vs_naive_pct: number;
  /** TCP conformal CI on the vol forecast [low, high] in pips. */
  confidence_interval_pips: [number, number];
  /** True if the model could not produce a forecast (uses naive ATR instead). */
  is_fallback: boolean;
}

// ─── H. Event readout ───────────────────────────────────────────────────────
export type Session =
  | 'asian'
  | 'london'
  | 'ny_overlap'
  | 'ny_afternoon'
  | 'after_hours';

export interface EventReadout {
  news_blackout_active: boolean;
  next_event_label: string | null;
  /** Minutes until the next HIGH-impact event. */
  next_event_in_minutes: number | null;
  /** Aggregated 24h sentiment in [-1, 1]. */
  sentiment_score: number;
  /** Mean confidence of constituent articles in [0, 1]. */
  sentiment_confidence: number;
  session: Session;
}

// ─── I. Component breakdown (8 components) ──────────────────────────────────
export type ComponentName =
  | 'bos'
  | 'fvg'
  | 'ob'
  | 'regime'
  | 'news'
  | 'volume'
  | 'momentum'
  | 'rsi_divergence';

export interface ComponentBreakdown {
  name: ComponentName;
  /** Effective weighted contribution to the final score. */
  contribution: number;
  /** Maximum allocable weight (may be redacted on B2C surfaces). */
  weight_max: number;
  /** Short human-readable rationale. */
  reasoning: string;
}

// ─── J. Historical statistics ───────────────────────────────────────────────
// Stats nullables tant que la validation OOS indépendante n'a pas franchi
// les gates de promotion (cf. AUDIT_ALGO_2026_05_27 + section Honnêteté).
// `backtest_window` reste toujours présent — soit fenêtre réelle, soit
// libellé « OOS validation pending — Sprint X ».
export interface HistoricalStats {
  /** Count of similar past setups in the backtest store. */
  similar_setups_n: number | null;
  /** Empirical win rate on similar setups [0, 1]. */
  hit_rate_observed: number | null;
  /** Profit factor = gross gains / gross losses. */
  profit_factor: number | null;
  /** 95% bootstrap CI for the profit factor [low, high]. */
  profit_factor_ci95: [number, number] | null;
  /** Conformal coverage observed on history (mirrors D.4). */
  empirical_coverage: number;
  /** Human label of the backtest window. */
  backtest_window: string;
}

// ─── K. Narrative & sources ─────────────────────────────────────────────────
export type NarrativeLanguage = 'fr' | 'en' | 'de' | 'es';

export type SourceType =
  | 'paper'
  | 'report'
  | 'data'
  | 'education'
  | 'internal';

export interface SourceCitation {
  type: SourceType;
  /** Canonical URL of the source. */
  ref: string;
  /** Human-readable label. */
  label: string;
  /** Optional quoted excerpt (≤ 500 chars). */
  quoted_excerpt?: string;
}

// ─── L. Compliance metadata ─────────────────────────────────────────────────
export interface ComplianceMeta {
  disclaimer_lang: NarrativeLanguage;
  /** ISO-3166 / OFAC list of blocked jurisdictions. */
  jurisdiction_blocked: string[];
  /** Honesty flag — stays false until empirical edge thresholds are crossed. */
  edge_claim: boolean;
  /** True while the system runs in paper-trading demonstration mode. */
  is_paper_demo: boolean;
}

// ─── Top-level signal ───────────────────────────────────────────────────────
export interface InsightSignalV2 {
  // Schema metadata (kept under a discriminator so future versions don't break consumers).
  schema_version: '2.1.0';

  // A. Identity
  id: string;
  instrument: Instrument;
  timeframe: Timeframe;
  created_at_utc: string;
  valid_until_utc: string;

  // B. Direction
  direction: Direction;

  // C. Conviction
  conviction_0_100: number;
  conviction_label: ConvictionLabel;

  // D. Uncertainty
  uncertainty: UncertaintyContext;

  // E-H. Readouts
  structure_readout: StructureReadout;
  regime_readout: RegimeReadout;
  volatility_readout: VolatilityReadout;
  event_readout: EventReadout;

  // I. Component breakdown
  breakdown_components: ComponentBreakdown[];

  // J. Historical stats (may be null on bootstrap signals)
  historical_stats: HistoricalStats | null;

  // K. Narrative
  narrative_short: string;
  narrative_long: string | null;
  narrative_language: NarrativeLanguage;
  sources_cited: SourceCitation[];

  // L. Compliance
  compliance: ComplianceMeta;
}

// ─── Convenience type guards & helpers ──────────────────────────────────────

export function isBullish(s: Pick<InsightSignalV2, 'direction'>): boolean {
  return s.direction === 'BULLISH_SETUP';
}

export function isBearish(s: Pick<InsightSignalV2, 'direction'>): boolean {
  return s.direction === 'BEARISH_SETUP';
}

export function isNeutral(s: Pick<InsightSignalV2, 'direction'>): boolean {
  return s.direction === 'NEUTRAL';
}
