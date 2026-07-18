import { useLocale, useTranslations } from 'next-intl';
import type {
  Direction,
  FVGStatus,
  ImpactLevel,
  LiquidityKind,
  LiquidityStatus,
  MarketPhase,
  MarketReadingHeader,
  MarketReadingStructure,
  OBImportance,
  OBStatus,
  RetestType,
  SurpriseDirection,
  TrendValue,
  ValidationStatus,
  VolatilityObserved,
} from '@/types/market-reading';
import type { Tone } from './formatters';
import { countActiveZones, deriveTrendMaturity } from './regime-facts';
import { MTF_TREND_ORDER, type MtfRelation, type MtfTrendMap } from './mtf-trend';

/**
 * Locale-aware reading formatters — the i18n counterpart of the enum-label and
 * relative-time helpers in `formatters.ts`. Labels come from the `reading.*`
 * message namespace; the tone/colour intent stays in code (language-neutral).
 * Number formatting (price, %) follows the active locale.
 *
 * `formatters.ts` keeps its pure functions (still used by not-yet-migrated
 * surfaces: RegimeSection/StructureSection prose, zones, chart, /app, chat).
 * All consumers of THIS hook are client components.
 */
const PRICE_DECIMALS: Record<string, number> = {
  XAUUSD: 2,
  EURUSD: 5,
  GBPUSD: 5,
  USDJPY: 3,
  US500: 1,
  BTCUSD: 2,
};

export function useReadingFormatters() {
  const t = useTranslations('reading');
  const locale = useLocale();

  const has = (key: string) => t.has(key);

  function instrument(code: string): string {
    return has(`labels.instrument_${code}`) ? t(`labels.instrument_${code}`) : code;
  }

  function timeframe(code: string): string {
    return has(`labels.timeframe_${code}`) ? t(`labels.timeframe_${code}`) : code;
  }

  function trend(v: TrendValue): { label: string; tone: Tone } {
    const tone: Tone = v === 'bullish' ? 'bull' : v === 'bearish' ? 'bear' : 'neutral';
    return { label: t(`labels.trend_${v}`), tone };
  }

  function volatility(v: VolatilityObserved): { label: string; tone: Tone } {
    return { label: t(`labels.volatility_${v}`), tone: v === 'elevated' ? 'warn' : 'neutral' };
  }

  function marketPhase(v: MarketPhase): { label: string; tone: Tone } {
    return { label: t(`labels.marketPhase_${v}`), tone: 'neutral' };
  }

  /** Compact one-word phase label (Régime badge « Phase : Tendance »). */
  function marketPhaseShort(v: MarketPhase): string {
    return t(`labels.marketPhaseShort_${v}`);
  }

  function impact(v: ImpactLevel): { label: string; tone: Tone } {
    return { label: t(`labels.impact_${v}`), tone: v === 'high' ? 'warn' : 'neutral' };
  }

  function surprise(v: SurpriseDirection): string {
    return t(`labels.surprise_${v}`);
  }

  function direction(v: Direction): string {
    return t(`labels.direction_${v}`);
  }

  function currency(code: string): string {
    return code.toUpperCase();
  }

  // ── Structure section labels (SMC vocabulary) ────────────────────────────────

  /** Feminine validation label (agrees with « cassure » / « structure »). */
  function validationFem(v: ValidationStatus): string {
    return t(`labels.validationFem_${v}`);
  }

  function obStatus(v: OBStatus): string {
    return t(`labels.obStatus_${v}`);
  }

  function obImportance(v: OBImportance): string {
    return t(`labels.obImportance_${v}`);
  }

  function fvgStatus(v: FVGStatus): string {
    return t(`labels.fvgStatus_${v}`);
  }

  function retestType(v: RetestType): string {
    return t(`labels.retestType_${v}`);
  }

  function liquidityKind(v: LiquidityKind): string {
    return t(`labels.liquidityKind_${v}`);
  }

  function liquidityStatus(v: LiquidityStatus): { label: string; tone: Tone } {
    return { label: t(`labels.liquidityStatus_${v}`), tone: v === 'swept' ? 'warn' : 'neutral' };
  }

  /** « 2 380,00 – 2 390,00 » price band, locale-aware. */
  function band(low: number, high: number, instrumentCode: string): string {
    return `${price(low, instrumentCode)} – ${price(high, instrumentCode)}`;
  }

  /**
   * Humanise a composite technical-trigger code (`<event>_<tf>[_<direction>]`),
   * e.g. "bos_h1_bullish" → "Cassure de structure haussier (H1)".
   */
  function triggerType(type: string): string {
    const parts = type.split('_');
    const event = parts[0] ?? '';
    const last = parts[parts.length - 1];
    const hasDir = last === 'bullish' || last === 'bearish';
    const dir = hasDir ? (last as Direction) : null;
    const tf = hasDir ? parts[parts.length - 2] : parts[parts.length - 1];
    const base = has(`labels.triggerEvent_${event}`) ? t(`labels.triggerEvent_${event}`) : type;
    const dirPart = dir ? ` ${direction(dir)}` : '';
    const tfPart = tf ? ` (${tf.toUpperCase()})` : '';
    return `${base}${dirPart}${tfPart}`;
  }

  // ── Numbers ────────────────────────────────────────────────────────────────

  function price(value: number, instrumentCode: string): string {
    const decimals = PRICE_DECIMALS[instrumentCode] ?? 2;
    return value.toLocaleString(locale, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }

  function changeTone(fraction: number | null | undefined): Tone {
    if (fraction === null || fraction === undefined || fraction === 0) return 'neutral';
    return fraction > 0 ? 'bull' : 'bear';
  }

  function changePercent(fraction: number): string {
    const pct = fraction * 100;
    const sign = pct > 0 ? '+' : pct < 0 ? '−' : '';
    const body = Math.abs(pct).toLocaleString(locale, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    return `${sign}${body} %`;
  }

  // ── Relative time ────────────────────────────────────────────────────────────

  const MINUTE = 60_000;
  const HOUR = 60 * MINUTE;
  const DAY = 24 * HOUR;

  /** "à l'instant", "il y a 12 minutes", "il y a 3 heures", "il y a 2 jours". */
  function relativePast(iso: string, now: Date = new Date()): string {
    const diff = now.getTime() - new Date(iso).getTime();
    if (diff < MINUTE) return t('time.instant');
    if (diff < HOUR) return t('time.relativePast', { n: Math.floor(diff / MINUTE) });
    if (diff < DAY) return t('time.relativePastHours', { n: Math.floor(diff / HOUR) });
    return t('time.relativePastDays', { n: Math.floor(diff / DAY) });
  }

  /** "dans 18 min", "dans 3h", "dans 2h05", "dans 2j" — upcoming events. */
  function timeToEvent(minutes: number): string {
    if (minutes <= 0) return t('time.toEventImminent');
    if (minutes < 60) return t('time.toEventMin', { n: Math.round(minutes) });
    if (minutes < 60 * 24) {
      const h = Math.floor(minutes / 60);
      const m = Math.round(minutes % 60);
      return m === 0
        ? t('time.toEventHours', { h })
        : t('time.toEventHoursMin', { h, mm: m.toString().padStart(2, '0') });
    }
    const d = Math.floor(minutes / (60 * 24));
    const remH = Math.floor((minutes % (60 * 24)) / 60);
    return remH === 0 ? t('time.toEventDays', { d }) : t('time.toEventDaysHours', { d, h: remH });
  }

  /** "il y a 5 min", "il y a 2h", "il y a 2h05" — recent triggers/news. */
  function minutesAgo(minutes: number): string {
    if (minutes <= 0) return t('time.instant');
    if (minutes < 60) return t('time.agoMin', { n: Math.round(minutes) });
    const h = Math.floor(minutes / 60);
    const m = Math.round(minutes % 60);
    return m === 0
      ? t('time.agoHours', { h })
      : t('time.agoHoursMin', { h, mm: m.toString().padStart(2, '0') });
  }

  // ── Régime section prose (calls the pure derivation in regime-facts/mtf-trend,
  //    builds the localized sentence here). ─────────────────────────────────────

  const dirOf = (v: TrendValue): 'up' | 'down' | 'flat' =>
    v === 'bullish' ? 'up' : v === 'bearish' ? 'down' : 'flat';

  /** « Structure orientée haussière depuis le CHOCH du 24/06 à 14:30 (≈ 18 bougies M15). » */
  function regimeMaturity(
    structure: MarketReadingStructure,
    header: MarketReadingHeader,
  ): string | null {
    const m = deriveTrendMaturity(structure, header);
    if (!m) return null;
    const orient = t(`labels.orient_${m.direction}`);
    const parsed = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})/.exec(m.brokenAt);
    const when = parsed
      ? t('regime.maturityWhen', {
          ts: t('regime.breakTimestamp', {
            day: parsed[3],
            month: parsed[2],
            hh: parsed[4],
            mm: parsed[5],
          }),
        })
      : '';
    const bars = m.bars != null ? t('regime.maturityBars', { n: m.bars, tf: header.timeframe }) : '';
    return t('regime.maturity', { orient, when, bars });
  }

  /** « CHOCH haussier confirmé (M15) » — most recent of structure.bos / choch. */
  function regimeLastEvent(
    structure: MarketReadingStructure,
    header: MarketReadingHeader,
  ): string | null {
    const { bos, choch } = structure;
    let kind: 'BOS' | 'CHOCH';
    let ev: NonNullable<typeof bos> | NonNullable<typeof choch>;
    if (bos && choch) {
      const newerIsBos =
        new Date(bos.broken_at).getTime() >= new Date(choch.broken_at).getTime();
      kind = newerIsBos ? 'BOS' : 'CHOCH';
      ev = newerIsBos ? bos : choch;
    } else if (bos) {
      kind = 'BOS';
      ev = bos;
    } else if (choch) {
      kind = 'CHOCH';
      ev = choch;
    } else {
      return null;
    }
    return t('regime.lastEvent', {
      kind,
      dir: t(`labels.direction_${ev.direction}`),
      val: t(`labels.validation_${ev.validation_status}`),
      tf: header.timeframe,
    });
  }

  /** « 1 OB · 2 FVG actifs » — count of active zones. */
  function regimeZoneDensity(structure: MarketReadingStructure): string {
    const { ob, fvg } = countActiveZones(structure);
    return t('regime.zoneDensity', { ob, fvg });
  }

  /**
   * Localized multi-timeframe alignment sentence. `kind` comes from the pure
   * `classifyMtfAlignment` (the RegimeSection already computes it for the
   * disagreement callout); this rebuilds the descriptive text per locale.
   */
  function mtfAlignmentText(trends: MtfTrendMap, kind: MtfRelation['kind']): string {
    const entries = MTF_TREND_ORDER.map(({ key, label }) => ({
      label: label as string,
      trend: trends[key],
    })).filter((e): e is { label: string; trend: TrendValue } => e.trend != null);
    const count = entries.length;
    if (count === 0 || kind === 'none') return '';
    if (kind === 'neutral') return t('regime.mtfNeutral', { count });
    if (kind === 'aligned') {
      const dir = dirOf(entries[0]!.trend) === 'up' ? 'up' : 'down';
      return t('regime.mtfAligned', { count, dir: t(`labels.alignedDir_${dir}`) });
    }
    if (kind === 'pullback') {
      const h4 = trends.h4;
      const fem = t(`labels.orient_${h4 && dirOf(h4) === 'up' ? 'bullish' : 'bearish'}`);
      return t('regime.mtfPullback', { fem });
    }
    // divergent | partial → list each TF's observed trend, joined per locale.
    const parts = entries.map((e) => `${e.label} ${t(`labels.trendAdj_${e.trend}`)}`);
    const list = new Intl.ListFormat(locale, { style: 'long', type: 'conjunction' }).format(parts);
    return t('regime.mtfDivergent', { list });
  }

  return {
    instrument,
    timeframe,
    trend,
    volatility,
    marketPhase,
    marketPhaseShort,
    impact,
    surprise,
    direction,
    currency,
    validationFem,
    obStatus,
    obImportance,
    fvgStatus,
    retestType,
    liquidityKind,
    liquidityStatus,
    band,
    triggerType,
    price,
    changeTone,
    changePercent,
    relativePast,
    timeToEvent,
    minutesAgo,
    regimeMaturity,
    regimeLastEvent,
    regimeZoneDensity,
    mtfAlignmentText,
  };
}
