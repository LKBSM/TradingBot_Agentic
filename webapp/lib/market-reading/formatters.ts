import type {
  Direction,
  FVGStatus,
  ImpactLevel,
  LiquidityKind,
  LiquiditySide,
  LiquidityStatus,
  MarketPhase,
  MTFBiasValue,
  MTFTimeframeKey,
  OBImportance,
  OBStatus,
  RetestType,
  SurpriseDirection,
  TrendValue,
  ValidationStatus,
  VolatilityObserved,
} from '@/types/market-reading';

/**
 * UI-facing strings derived from the raw MarketReading contract. Kept in a
 * single module so wording is auditable in one place (compliance reviews, FR
 * adjustments, future i18n). Everything user-visible lands here, not inlined in
 * JSX.
 *
 * Niveau 1.5 strict: every label DESCRIBES a market fact. No directive verbs
 * (acheter / vendre / signal directif), no synthetic conviction score.
 */

// ─── Tone (shared colour intent) ─────────────────────────────────────────────

export type Tone = 'bull' | 'bear' | 'neutral' | 'warn';

// ─── Instrument / timeframe ──────────────────────────────────────────────────

const INSTRUMENT_LABEL: Record<string, string> = {
  XAUUSD: 'Or (XAU/USD)',
  EURUSD: 'Euro / Dollar (EUR/USD)',
  BTCUSD: 'Bitcoin (BTC/USD)',
  US500: 'S&P 500 (US500)',
  GBPUSD: 'Livre / Dollar (GBP/USD)',
  USDJPY: 'Dollar / Yen (USD/JPY)',
};

export function formatInstrument(instrument: string): string {
  return INSTRUMENT_LABEL[instrument] ?? instrument;
}

const TIMEFRAME_LABEL: Record<string, string> = {
  M1: '1 minute',
  M5: '5 minutes',
  M15: '15 minutes',
  M30: '30 minutes',
  H1: '1 heure',
  H4: '4 heures',
  D1: '1 jour',
  W1: '1 semaine',
};

export function formatTimeframe(timeframe: string): string {
  return TIMEFRAME_LABEL[timeframe] ?? timeframe;
}

// ─── Price ────────────────────────────────────────────────────────────────────

const PRICE_DECIMALS: Record<string, number> = {
  XAUUSD: 2,
  EURUSD: 5,
  GBPUSD: 5,
  USDJPY: 3,
  US500: 1,
  BTCUSD: 2,
};

/** Format a price using the instrument's conventional precision (default 2). */
export function formatPrice(value: number, instrument: string): string {
  const decimals = PRICE_DECIMALS[instrument] ?? 2;
  return value.toLocaleString('fr-FR', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/** Format a [low, high] price band. */
export function formatBand(
  low: number,
  high: number,
  instrument: string,
): string {
  return `${formatPrice(low, instrument)} – ${formatPrice(high, instrument)}`;
}

/**
 * Format a fractional change as a signed percentage, fr-FR ("-3,22 %",
 * "+1,10 %", "0,00 %"). Descriptive market fact, not a forecast.
 */
export function formatChangePercent(fraction: number): string {
  const pct = fraction * 100;
  const sign = pct > 0 ? '+' : pct < 0 ? '−' : '';
  const body = Math.abs(pct).toLocaleString('fr-FR', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  return `${sign}${body} %`;
}

/** Colour intent for a price change (green up / red down / muted flat). */
export function changeTone(fraction: number | null | undefined): Tone {
  if (fraction === null || fraction === undefined || fraction === 0) {
    return 'neutral';
  }
  return fraction > 0 ? 'bull' : 'bear';
}

// ─── Time ──────────────────────────────────────────────────────────────────────

const SECOND = 1_000;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/** "à l'instant", "il y a 12 minutes", "il y a 3 heures", "il y a 2 jours". */
export function formatRelativePast(iso: string, now: Date = new Date()): string {
  const diff = now.getTime() - new Date(iso).getTime();
  if (diff < MINUTE) return "à l'instant";
  if (diff < HOUR) {
    const m = Math.floor(diff / MINUTE);
    return `il y a ${m} minute${m > 1 ? 's' : ''}`;
  }
  if (diff < DAY) {
    const h = Math.floor(diff / HOUR);
    return `il y a ${h} heure${h > 1 ? 's' : ''}`;
  }
  const d = Math.floor(diff / DAY);
  return `il y a ${d} jour${d > 1 ? 's' : ''}`;
}

/** "dans 18 min", "dans 3h", "dans 2h05", "dans 2j" — for upcoming events. */
export function formatTimeToEvent(minutes: number): string {
  if (minutes <= 0) return 'imminent';
  if (minutes < 60) return `dans ${Math.round(minutes)} min`;
  if (minutes < 60 * 24) {
    const h = Math.floor(minutes / 60);
    const m = Math.round(minutes % 60);
    return m === 0 ? `dans ${h}h` : `dans ${h}h${m.toString().padStart(2, '0')}`;
  }
  const d = Math.floor(minutes / (60 * 24));
  const remH = Math.floor((minutes % (60 * 24)) / 60);
  return remH === 0 ? `dans ${d}j` : `dans ${d}j${remH}h`;
}

/** "il y a 5 min", "il y a 2h" — for recent technical triggers / news. */
export function formatMinutesAgo(minutes: number): string {
  if (minutes <= 0) return "à l'instant";
  if (minutes < 60) return `il y a ${Math.round(minutes)} min`;
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes % 60);
  return m === 0 ? `il y a ${h}h` : `il y a ${h}h${m.toString().padStart(2, '0')}`;
}

// ─── Regime: trend / volatility / market phase (MarketPhasePanel) ────────────

const TREND_LABEL: Record<TrendValue, string> = {
  bullish: 'Tendance haussière',
  bearish: 'Tendance baissière',
  neutral: 'Tendance neutre',
  ranging: 'Marché en range',
};

export function formatTrend(trend: TrendValue): { label: string; tone: Tone } {
  const tone: Tone =
    trend === 'bullish' ? 'bull' : trend === 'bearish' ? 'bear' : 'neutral';
  return { label: TREND_LABEL[trend], tone };
}

const VOLATILITY_LABEL: Record<VolatilityObserved, string> = {
  low: 'Volatilité basse',
  normal: 'Volatilité normale',
  elevated: 'Volatilité élevée',
};

export function formatVolatility(v: VolatilityObserved): {
  label: string;
  tone: Tone;
} {
  const tone: Tone = v === 'elevated' ? 'warn' : 'neutral';
  return { label: VOLATILITY_LABEL[v], tone };
}

const MARKET_PHASE_LABEL: Record<MarketPhase, string> = {
  accumulation: 'Phase d’accumulation',
  distribution: 'Phase de distribution',
  trend: 'Phase de tendance',
  ranging: 'Phase de range',
  expansion: 'Phase d’expansion',
};

export function formatMarketPhase(phase: MarketPhase): {
  label: string;
  tone: Tone;
} {
  return { label: MARKET_PHASE_LABEL[phase], tone: 'neutral' };
}

// Compact one-word phase label for the Régime section badge (« Phase : Tendance »),
// distinct from the hero MarketPhasePanel's full « Phase de tendance » wording.
const MARKET_PHASE_SHORT: Record<MarketPhase, string> = {
  accumulation: 'Accumulation',
  distribution: 'Distribution',
  trend: 'Tendance',
  ranging: 'Range',
  expansion: 'Expansion',
};

export function formatMarketPhaseShort(phase: MarketPhase): string {
  return MARKET_PHASE_SHORT[phase];
}

// ─── MTF confluence ──────────────────────────────────────────────────────────

const MTF_BIAS_LABEL: Record<MTFBiasValue, string> = {
  bullish: 'haussier',
  bearish: 'baissier',
  neutral: 'neutre',
  ranging: 'range',
};

export function formatMtfBias(bias: MTFBiasValue): { label: string; tone: Tone } {
  const tone: Tone =
    bias === 'bullish' ? 'bull' : bias === 'bearish' ? 'bear' : 'neutral';
  return { label: MTF_BIAS_LABEL[bias], tone };
}

export function formatMtfKey(key: MTFTimeframeKey): string {
  return key.toUpperCase();
}

// ─── Structure: SMC vocabulary ────────────────────────────────────────────────

const DIRECTION_LABEL: Record<Direction, string> = {
  bullish: 'haussier',
  bearish: 'baissier',
};

export function formatDirection(direction: Direction): string {
  return DIRECTION_LABEL[direction];
}

const VALIDATION_LABEL: Record<ValidationStatus, string> = {
  confirmed: 'confirmée',
  pending: 'en attente de confirmation',
  invalidated: 'invalidée',
};

export function formatValidationStatus(status: ValidationStatus): string {
  return VALIDATION_LABEL[status];
}

const OB_STATUS_LABEL: Record<OBStatus, string> = {
  active: 'actif',
  mitigated: 'mitigé',
  invalidated: 'invalidé',
};

export function formatObStatus(status: OBStatus): string {
  return OB_STATUS_LABEL[status];
}

const OB_IMPORTANCE_LABEL: Record<OBImportance, string> = {
  low: 'faible',
  medium: 'moyenne',
  high: 'élevée',
};

export function formatObImportance(importance: OBImportance): string {
  return OB_IMPORTANCE_LABEL[importance];
}

const FVG_STATUS_LABEL: Record<FVGStatus, string> = {
  active: 'active',
  partially_filled: 'partiellement comblée',
  filled: 'comblée',
};

export function formatFvgStatus(status: FVGStatus): string {
  return FVG_STATUS_LABEL[status];
}

const RETEST_TYPE_LABEL: Record<RetestType, string> = {
  bos_retest: 'retest de cassure (BOS)',
  choch_retest: 'retest de changement de caractère (CHOCH)',
  ob_retest: 'retest d’Order Block',
  fvg_retest: 'retest de Fair Value Gap',
};

export function formatRetestType(type: RetestType): string {
  return RETEST_TYPE_LABEL[type];
}

// ─── Structure: external liquidity ────────────────────────────────────────────

const LIQUIDITY_SIDE_LABEL: Record<LiquiditySide, string> = {
  bsl: 'liquidité acheteuse (au-dessus)',
  ssl: 'liquidité vendeuse (en-dessous)',
};

export function formatLiquiditySide(side: LiquiditySide): string {
  return LIQUIDITY_SIDE_LABEL[side];
}

/** Short axis/badge code — `BSL` (buy-side) / `SSL` (sell-side). */
const LIQUIDITY_SIDE_SHORT: Record<LiquiditySide, string> = {
  bsl: 'BSL',
  ssl: 'SSL',
};

export function formatLiquiditySideShort(side: LiquiditySide): string {
  return LIQUIDITY_SIDE_SHORT[side];
}

const LIQUIDITY_KIND_LABEL: Record<LiquidityKind, string> = {
  equal_highs: 'sommets égaux',
  equal_lows: 'creux égaux',
  range_high: 'extrême haut de range',
  range_low: 'extrême bas de range',
};

export function formatLiquidityKind(kind: LiquidityKind): string {
  return LIQUIDITY_KIND_LABEL[kind];
}

const LIQUIDITY_STATUS_LABEL: Record<LiquidityStatus, { label: string; tone: Tone }> = {
  intact: { label: 'intacte', tone: 'neutral' },
  swept: { label: 'prise', tone: 'warn' },
  broken: { label: 'cassée', tone: 'neutral' },
};

export function formatLiquidityStatus(status: LiquidityStatus): {
  label: string;
  tone: Tone;
} {
  return LIQUIDITY_STATUS_LABEL[status];
}

// ─── Events ────────────────────────────────────────────────────────────────────

const IMPACT_LABEL: Record<ImpactLevel, { label: string; tone: Tone }> = {
  low: { label: 'Impact faible', tone: 'neutral' },
  medium: { label: 'Impact moyen', tone: 'neutral' },
  high: { label: 'Impact fort', tone: 'warn' },
};

export function formatImpact(impact: ImpactLevel): { label: string; tone: Tone } {
  return IMPACT_LABEL[impact];
}

const SURPRISE_LABEL: Record<SurpriseDirection, string> = {
  beat: 'au-dessus du consensus',
  miss: 'en-dessous du consensus',
  in_line: 'conforme au consensus',
};

export function formatSurprise(surprise: SurpriseDirection): string {
  return SURPRISE_LABEL[surprise];
}

/**
 * Humanise a composite technical-trigger code
 * (`<event>_<tf>[_<direction>]`) into plain French, e.g.
 * "bos_h1_bullish" → "Cassure de structure haussière (H1)".
 */
export function formatTriggerType(type: string): string {
  const parts = type.split('_');
  const event = parts[0];
  // Direction (bullish/bearish) only present for bos/choch; tf is the last
  // non-direction token.
  const hasDirection =
    parts[parts.length - 1] === 'bullish' ||
    parts[parts.length - 1] === 'bearish';
  const direction = hasDirection ? parts[parts.length - 1] : null;
  const tf = hasDirection ? parts[parts.length - 2] : parts[parts.length - 1];

  const EVENT: Record<string, string> = {
    bos: 'Cassure de structure',
    choch: 'Changement de caractère',
    ob: 'Mitigation d’Order Block',
    fvg: 'Comblement de Fair Value Gap',
    retest: 'Retest',
  };
  const base = EVENT[event ?? ''] ?? type;
  const dir =
    direction === 'bullish'
      ? ' haussier'
      : direction === 'bearish'
        ? ' baissier'
        : '';
  const tfLabel = tf ? ` (${tf.toUpperCase()})` : '';
  return `${base}${dir}${tfLabel}`;
}

// ─── Currency ──────────────────────────────────────────────────────────────────

export function formatCurrency(code: string): string {
  return code.toUpperCase();
}
