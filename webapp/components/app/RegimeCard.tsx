'use client';

import * as React from 'react';
import { useLocale, useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import { useMtfTrends } from '@/lib/market-reading/hooks';
import { mtfOrderFor } from '@/lib/market-reading/mtf-trend';
import { isSessionRelevant, isPrevLevelsRelevant } from '@/lib/timeframes';
import {
  deriveTrendMaturity,
  trendWindowSpan,
  TREND_WINDOW_BARS,
} from '@/lib/market-reading/regime-facts';
import { formatLocalDayLong, formatLocalHm, parseUtc } from '@/lib/time/localTime';
import {
  structureRange,
  positionPct,
  referenceLevelsFromPayload,
  distancePct,
  type ReferenceLevels,
} from '@/lib/market-reading/reference-levels';
import {
  matchLiquidity,
  coincidenceTolerance,
} from '@/lib/market-reading/reference-coincidence';
import { formatLiquiditySideShort } from '@/lib/market-reading/formatters';
import { useChartViewOptional } from '@/lib/chart/viewState';
import {
  computeSession,
  splitDelay,
  formatWeeklyClose,
  type SessionInfo,
  type SessionName,
  type TransitionLabel,
} from '@/lib/market-reading/sessions';
import type {
  BOSRecent,
  CHOCHRecent,
  LiquiditySide,
  MarketReadingHeader,
  MarketReadingRegime,
  MarketReadingStructure,
  MarketStatusPayload,
  ReferenceLevelsPayload,
} from '@/types/market-reading';
import './regime-tiles.css';

// ─── openHelp coordination (one panel open across the whole page) ─────────────
// The regime detail panel rides the page-wide `openHelp` string so opening it
// closes any other card's help and vice-versa. Encoding: `rg:<tileKey>:<tab>`.
const RG = 'rg:';
type Tab = 'data' | 'concept';
/** The six numeric reference-level keys (excludes the completeness booleans). */
type LevelKey =
  | 'dayOpen'
  | 'weekOpen'
  | 'prevDayHigh'
  | 'prevDayLow'
  | 'prevWeekHigh'
  | 'prevWeekLow';
function parseOpen(openHelp: string | null): { key: string; tab: Tab } | null {
  if (!openHelp || !openHelp.startsWith(RG)) return null;
  const parts = openHelp.split(':');
  const key = parts[1];
  const tab = parts[2];
  if (!key || (tab !== 'data' && tab !== 'concept')) return null;
  return { key, tab };
}

function withBold(text: string): React.ReactNode {
  return text.split('**').map((seg, i) =>
    i % 2 === 1 ? <b key={i}>{seg}</b> : <React.Fragment key={i}>{seg}</React.Fragment>,
  );
}

function arrowOf(dir: 'bullish' | 'bearish' | 'indeterminate' | null): string {
  // TR-1: indeterminate (– no structural trend) and unavailable (· no reading)
  // read differently — neither is ever counted as an agreement.
  return dir === 'bullish'
    ? '↑'
    : dir === 'bearish'
      ? '↓'
      : dir === 'indeterminate'
        ? '–'
        : '·';
}

function latestBreak<T extends BOSRecent | CHOCHRecent>(
  events: T[] | undefined,
  fallback: T | null | undefined,
): T | null {
  let best: T | null = null;
  for (const e of events ?? []) {
    if (!best || new Date(e.broken_at).getTime() > new Date(best.broken_at).getTime()) best = e;
  }
  return best ?? fallback ?? null;
}

// ─── Small presentational helpers for the Donnée tab ──────────────────────────
function Dh4({ children, mt }: { children: React.ReactNode; mt?: boolean }) {
  return <div className={cn('dh4', mt && 'mt')}>{children}</div>;
}
function Dp({ children }: { children: React.ReactNode }) {
  return <p className="dp">{children}</p>;
}
function EvRow({ k, v, t }: { k: React.ReactNode; v?: React.ReactNode; t?: React.ReactNode }) {
  return (
    <div className="ev-r">
      <span className="ev-k">{k}</span>
      {v != null && <span className="ev-v">{v}</span>}
      {t != null && <span className="ev-t">{t}</span>}
    </div>
  );
}

/**
 * A clickable price → traces it as a calendar reference line on the chart, with
 * an explicit label (« <name> · <price> »), and brings the view to it. Re-click
 * removes it; a single trace at a time (RG-1c part C — every displayed price
 * LEVEL is traceable with the same gesture; deltas/averages are not levels and
 * stay plain text). `children` overrides the button face (else the formatted
 * price).
 */
function PxBtn({
  label,
  value,
  c,
  children,
}: {
  label: string;
  value: number;
  c: DataCtx;
  children?: React.ReactNode;
}) {
  const display = c.refLabel(label, value);
  const on =
    c.referenceLevel != null &&
    c.referenceLevel.price === value &&
    c.referenceLevel.label === display;
  return (
    <button
      type="button"
      className={cn('pxbtn', on && 'on')}
      aria-pressed={on}
      onClick={() => c.traceLevel(label, value)}
    >
      {children ?? c.px(value)}
    </button>
  );
}

/**
 * VZ-1b — a clickable EVENT (BOS/CHOCH) row value. Same affordance as PxBtn, but
 * it selects the event on the temporal channel so the camera frames its
 * confirmation BAR (not just its price). Shows the broken level as its face.
 */
function EvBtn({
  kind,
  ev,
  c,
}: {
  kind: 'bos' | 'choch';
  ev: { broken_at: string; direction: 'bullish' | 'bearish'; level: number };
  c: DataCtx;
}) {
  const d = parseUtc(ev.broken_at);
  const id = d ? `${kind}:${Math.floor(d.getTime() / 1000)}:${ev.level}` : null;
  const on = id != null && c.selectedEventId === id;
  return (
    <button
      type="button"
      className={cn('pxbtn', on && 'on')}
      aria-pressed={on}
      onClick={() => c.selectEvent(kind, ev)}
    >
      {c.px(ev.level)}
    </button>
  );
}

interface RegimeCardProps {
  regime: MarketReadingRegime;
  structure: MarketReadingStructure;
  header: MarketReadingHeader;
  price: number;
  /**
   * Epoch SECONDS of the candle/tick that produced `price` (liveHeader.priceTs).
   * Surfaces the « Prix courant » freshness so an apparent gap with the chart's
   * live line reads as elapsed time, not an error. Optional (null on fixtures).
   */
  priceTs?: number | null;
  /** Raw server market status — carries the session windows (single source). */
  marketStatus: MarketStatusPayload | null;
  /** Server-aggregated calendar reference levels (RG-1c) — day/week open + prev
   *  extremes over the MC-1 trading calendar. Null on static fixtures. */
  referenceLevelsPayload: ReferenceLevelsPayload | null;
  openHelp: string | null;
  onToggleHelp: (key: string) => void;
}

interface Tile {
  key: string;
  label: string;
  /** Facade body: either a plain value (+ optional mono) or a custom node. */
  value?: string | null;
  mono?: boolean;
  html?: React.ReactNode;
  sub: string | null;
  /** A measure with no engine datum is dropped entirely (no tile). */
  available: boolean;
}

/**
 * "Régime de marché" — ten descriptive measures, each backed by real engine
 * output (mission RG-1). A tile opens a two-tab detail panel: « Donnée » (the
 * live values that produced the figure — the proof) and « Concept » (fixed
 * pedagogy, incl. a mandatory « ce que ça ne dit pas » block). The reference-
 * level tile traces a calendar marker on the chart through a DEDICATED channel
 * that never touches the zone id-lock. No score is ever combined. A measure the
 * engine can't source is not rendered at all — never « N/A », never 0.
 */
export function RegimeCard({
  regime,
  structure,
  header,
  price,
  priceTs = null,
  marketStatus,
  referenceLevelsPayload,
  openHelp,
  onToggleHelp,
}: RegimeCardProps) {
  const t = useTranslations('regimePanel');
  const tr = useTranslations('reading');
  const locale = useLocale();
  const fmt = useReadingFormatters();
  const instrument = header.instrument;
  const tf = header.timeframe;
  const { trends } = useMtfTrends(instrument, tf);
  // The units ABOVE the viewed one (TF-1 decision C) — relative, not a fixed
  // H4·H1·M15 triplet. Empty only at the very top of the ladder.
  const mtfOrder = mtfOrderFor(tf);
  // TF-1 decision D — on a unit whose candle spans a whole day (D1/W1), Session
  // and "veille" reference levels are not applicable → hidden WITH a mention.
  const sessionRelevant = isSessionRelevant(tf);
  const prevLevelsRelevant = isPrevLevelsRelevant(tf);
  const { referenceLevel, setReferenceLevel, selection, select, clearSelection } =
    useChartViewOptional();

  // Server-aggregated over the MC-1 trading calendar (RG-1c) — the front never
  // indexes a single candle to build these.
  const refLevels: ReferenceLevels | null = referenceLevelsFromPayload(referenceLevelsPayload);

  const px = (v: number) => fmt.price(v, instrument);
  // « Prix courant » freshness — the EXACT local time the displayed price was
  // read (from its epoch `priceTs`). Deterministic given priceTs, so it renders
  // straight (RegimeCard is client-rendered over client-fetched data).
  const priceTimeLabel =
    priceTs != null && Number.isFinite(priceTs)
      ? tr('temporal.priceAt', { time: formatLocalHm(new Date(priceTs * 1000)) })
      : null;
  // Long, localized date — « 24 juil. à 09:45 » (fr) / « Jul 24 at 09:45 » (en),
  // never the ambiguous numeric « 24/07 ».
  const dayHm = (iso: string) => {
    const d = parseUtc(iso);
    return d ? formatLocalDayLong(d, locale) : '';
  };

  // ── Derived facts (all read-only over engine output) ────────────────────────
  const maturity = deriveTrendMaturity(structure, header);

  // TR-1: the Trend tile names the structural event it rides on (« depuis le
  // CHOCH haussier du 24 juil. ») so it tells the SAME story as the Maturité
  // tile; when indeterminate it says so, never a manufactured window.
  const tref = regime.trend_reference ?? null;
  const trendSub = tref
    ? t('sub.trendRef', {
        kind: tref.kind === 'choch' ? t('value.kindChoch') : t('value.kindBos'),
        dir: tref.direction === 'bullish' ? t('value.dirUp') : t('value.dirDown'),
        date: dayHm(tref.broken_at),
      })
    : t('sub.trendNone');

  const avail = mtfOrder.filter(({ key }) => trends[key] != null);
  // TR-1: only units with a DETERMINATE structural trend form the alignment
  // denominator. Indeterminate units are counted apart (never agreement nor
  // disagreement) and surfaced with their own glyph — the ratio is « N sur M »
  // where M excludes them, and M is shown so the adjustment is visible.
  const directional = avail.filter(
    ({ key }) => trends[key] === 'bullish' || trends[key] === 'bearish',
  );
  const indeterminateCount = avail.length - directional.length;
  const dirs = directional.map(({ key }) => (trends[key] === 'bullish' ? 'up' : 'down'));
  const up = dirs.filter((d) => d === 'up').length;
  const down = dirs.filter((d) => d === 'down').length;
  let domArrow = '–';
  let domKind: 'up' | 'down' | 'flat' = 'flat';
  let aligned = 0;
  if (directional.length > 0 && up >= down) {
    domArrow = '↑';
    domKind = 'up';
    aligned = up;
  } else if (directional.length > 0) {
    domArrow = '↓';
    domKind = 'down';
    aligned = down;
  }

  const lc = latestBreak(structure.choch_events, structure.current_choch);
  const lb = latestBreak(structure.bos_events, structure.current_bos);
  let last: (BOSRecent | CHOCHRecent) | null = null;
  let lastKind: 'CHOCH' | 'BOS' | null = null;
  if (lc && lb) {
    const cNewer = new Date(lc.broken_at).getTime() >= new Date(lb.broken_at).getTime();
    last = cNewer ? lc : lb;
    lastKind = cNewer ? 'CHOCH' : 'BOS';
  } else if (lc) {
    last = lc;
    lastKind = 'CHOCH';
  } else if (lb) {
    last = lb;
    lastKind = 'BOS';
  }

  // Density by state (partially-filled FVGs count as OPEN, per the mission).
  const obActive = structure.order_blocks.filter((z) => z.status === 'active');
  const obTested = obActive.filter((z) => z.tested);
  const obMitigated = structure.order_blocks.filter((z) => z.status === 'mitigated');
  const fvgActive = structure.fair_value_gaps.filter((z) => z.status === 'active');
  const fvgPartial = structure.fair_value_gaps.filter((z) => z.status === 'partially_filled');
  const fvgOpen = fvgActive.length + fvgPartial.length;

  // Position in range.
  const range = structureRange(structure);
  const pos = range ? positionPct(range.low, range.high, price) : null;

  const vd = regime.volatility_detail;

  // Session — current session / next transition / overlap, computed live from
  // the server's session windows (single source: MC-1). Re-snapshot on tick.
  const session = React.useMemo(
    () => computeSession(marketStatus, new Date()),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [marketStatus, price],
  );

  const sessionName = (name: SessionName): string => t(`session.${name}`);
  const transitionName = (label: TransitionLabel): string => t(`session.${label}`);
  const fmtDelay = (mins: number): string => {
    const { d, h, m } = splitDelay(mins);
    if (d > 0) return t('delay.dh', { d, h });
    if (h > 0) return t('delay.hm', { h, mm: String(m).padStart(2, '0') });
    return t('delay.m', { m });
  };
  const sessionSub = (s: SessionInfo): string | null => {
    const localPart = s.localTime ? ` · ${s.localTime} NY` : '';
    if (s.next) {
      return (
        t('sub.sess', { label: transitionName(s.next.label), delay: fmtDelay(s.next.inMinutes) }) +
        localPart
      );
    }
    return s.localTime ? `${s.localTime} NY` : null;
  };

  // Reference-level rows (measure 10), each present only if computable.
  const levelRows: { key: LevelKey; label: string; value: number }[] = [];
  if (refLevels) {
    const order: [LevelKey, string][] = [
      ['dayOpen', t('data.dayOpen')],
      ['weekOpen', t('data.weekOpen')],
      ['prevDayHigh', t('data.prevDayHigh')],
      ['prevDayLow', t('data.prevDayLow')],
      ['prevWeekHigh', t('data.prevWeekHigh')],
      ['prevWeekLow', t('data.prevWeekLow')],
    ];
    for (const [k, label] of order) {
      const v = refLevels[k];
      if (v != null) levelRows.push({ key: k, label, value: v });
    }
  }
  // Periods the cached candles could not fully cover — the tile still shows the
  // levels it CAN source and names « données insuffisantes » for the rest, never
  // a partial value presented as complete (RG-1c).
  const levelsInsufficient: string[] = [];
  if (refLevels) {
    if (!refLevels.dayComplete) levelsInsufficient.push(t('data.lvlPeriodDay'));
    if (!refLevels.weekComplete) levelsInsufficient.push(t('data.lvlPeriodWeek'));
  }

  // ── Tile facades ────────────────────────────────────────────────────────────
  const tiles: Tile[] = [
    // Phase de marché — SUPPRIMÉE (RG-1b) : le moteur n'expose aucune phase
    // fondée sur le franchissement des bornes de structure ; `market_phase` n'est
    // qu'une dérivation de la tendance + volatilité (déjà deux tuiles). Pas de
    // donnée réelle → pas de tuile (jamais une dérivation). Ajout moteur possible.
    {
      key: 'trend',
      label: t('tiles.trend'),
      value: fmt.trend(regime.trend).label,
      sub: trendSub,
      available: true,
    },
    {
      key: 'vol',
      label: t('tiles.vol'),
      value: fmt.volatility(regime.volatility_observed).label,
      // Names the real denominator (« 7 dernières vs 20 précédentes ») from the
      // engine's volatility_detail — a measure without its denominator is unverifiable.
      sub: t('sub.vol', { recent: vd?.recent_n ?? 7, baseline: vd?.baseline_n ?? 20 }),
      available: true,
    },
    {
      key: 'pos',
      label: t('tiles.pos'),
      value: pos != null ? t('value.pos', { pct: Math.round(pos) }) : null,
      mono: true,
      sub: range ? t('sub.pos', { low: px(range.low), high: px(range.high) }) : null,
      available: pos != null && range != null,
    },
    {
      key: 'align',
      label: t('tiles.align'),
      value:
        mtfOrder.length === 0
          ? t('value.alignNone')
          : directional.length > 0
            ? t('value.align', { count: aligned, total: directional.length, arrow: domArrow })
            : avail.length > 0
              ? t('value.alignIndeterminate')
              : null,
      mono: true,
      // Name every upper unit; indeterminate shows –, unavailable shows · — TR-1:
      // neither is ever counted as an agreement. When some units are
      // indeterminate, name how many so the « N sur M » denominator is honest.
      sub:
        mtfOrder.length > 0
          ? mtfOrder.map(({ key, label }) => `${label} ${arrowOf(trends[key] ?? null)}`).join(' · ') +
            (indeterminateCount > 0
              ? ` · ${t('value.alignIndetCount', { count: indeterminateCount })}`
              : '')
          : null,
      available: mtfOrder.length === 0 || avail.length > 0,
    },
    {
      key: 'mat',
      label: t('tiles.mat'),
      value:
        maturity?.bars != null
          ? t('value.mat', {
              count: maturity.bars,
              // « ≈ » only when the count is a wall-clock estimate (no bars_ago on
              // the event); empty when it is the engine's real analysed-bar count.
              approx: maturity.barsApproximate ? '≈ ' : '',
            })
          : null,
      mono: true,
      // Name the CHOCH DIRECTION next to the date so the anchor's orientation is
      // legible right beside « Tendance » — a bullish CHOCH under a bearish trend
      // is no longer silent (DG-1 point 2 / display incoherence D-1).
      sub: maturity
        ? t('sub.mat', {
            dir: maturity.direction === 'bullish' ? t('value.dirUp') : t('value.dirDown'),
            date: dayHm(maturity.brokenAt),
          })
        : null,
      available: maturity != null && maturity.bars != null,
    },
    {
      key: 'last',
      label: t('tiles.last'),
      value: last ? `${lastKind} ${last.direction === 'bullish' ? '↑' : '↓'}` : null,
      sub: last ? dayHm(last.broken_at) : null,
      available: last != null,
    },
    {
      key: 'dens',
      label: t('tiles.dens'),
      value: t('value.dens', { ob: obActive.length, fvg: fvgOpen }),
      mono: true,
      sub: t('sub.dens', { tf }),
      available: true,
    },
    {
      key: 'sess',
      // VALUE = the current session (not the market state — the header badge
      // already shows open/closed; no value lives in two places).
      label: t('tiles.sess'),
      value: !sessionRelevant
        ? t('value.notApplicable')
        : session
          ? sessionName(session.current)
          : null,
      sub: !sessionRelevant ? t('sub.sessNA', { tf }) : session ? sessionSub(session) : null,
      available: !sessionRelevant || session != null,
    },
    {
      key: 'lvl',
      label: t('tiles.lvl'),
      value: !prevLevelsRelevant ? t('value.notApplicable') : undefined,
      html:
        prevLevelsRelevant && levelRows.length > 0 ? (
          <>
            {levelRows.slice(2, 4).map((r) => (
              <div className="lvlmini" key={r.key}>
                <span>{r.label}</span>
                <b>{px(r.value)}</b>
              </div>
            ))}
          </>
        ) : null,
      sub: !prevLevelsRelevant
        ? t('sub.lvlNA', { tf })
        : levelRows.length > 0
          ? t('sub.lvl', { count: levelRows.length })
          : null,
      available: !prevLevelsRelevant || levelRows.length > 0,
    },
  ];

  const shown = tiles.filter((tile) => tile.available);
  const open = parseOpen(openHelp);
  const openTile = open ? tiles.find((tile) => tile.key === open.key) : null;
  // The global « ? » (regime) is conceptual only — no Donnée tab.
  const isRegime = open?.key === 'regime';
  const activeTab: Tab = isRegime ? 'concept' : (open?.tab ?? 'data');

  const panelTitle = isRegime ? tr('regime.title') : (openTile?.label ?? tr('regime.title'));

  function toggle(key: string, tab: Tab) {
    onToggleHelp(`${RG}${key}:${tab}`);
  }

  // Coincidence of a repère with a DETECTED liquidity pocket (RG-1d). Tolerance =
  // a quarter of the recent average candle amplitude; null when no basis → no
  // coincidence claimed. Verified against real engine pools, never assumed — a
  // « haut de la veille » is NOT a BSL, it may merely COINCIDE with one.
  const coincidenceTol = coincidenceTolerance(regime.volatility_detail?.recent_avg);
  const coincidentSide = React.useCallback(
    (value: number): LiquiditySide | null =>
      matchLiquidity(value, structure.liquidity_pools, coincidenceTol),
    [structure.liquidity_pools, coincidenceTol],
  );

  // Explicit chart-line label — « Haut de la veille · 4 202,03 », prefixed with the
  // pocket side when the level coincides with a detected pocket (« BSL · Haut de la
  // veille · … ») — so a calendar repère painted on the chart is never mistaken for
  // a detected zone (RG-1c) and a real level coincidence is named (RG-1d).
  const refLabel = (label: string, value: number) => {
    const side = coincidentSide(value);
    const named = side ? `${formatLiquiditySideShort(side)} · ${label}` : label;
    return `${named} · ${px(value)}`;
  };

  function traceLevel(label: string, value: number) {
    const display = refLabel(label, value);
    const same =
      referenceLevel != null && referenceLevel.price === value && referenceLevel.label === display;
    setReferenceLevel(same ? null : { price: value, label: display });
  }

  // VZ-1b — an EVENT (BOS/CHOCH) is a couple date+prix : it selects on the unified
  // selection's EVENT channel (temporal frame on the confirmation candle via
  // frameEvent), NOT as a price-only reference level. Distinct path from the zone
  // id-lock; carries broken_at → epoch so the camera goes to the RIGHT bar.
  const selectedEventId = selection?.family === 'event' ? selection.id : null;
  function selectEvent(
    kind: 'bos' | 'choch',
    ev: { broken_at: string; direction: 'bullish' | 'bearish'; level: number },
  ) {
    const d = parseUtc(ev.broken_at);
    if (!d) return; // no confirmation timestamp → no honest temporal frame.
    const atSec = Math.floor(d.getTime() / 1000);
    const id = `${kind}:${atSec}:${ev.level}`;
    if (selectedEventId === id) {
      clearSelection();
      return;
    }
    select({ family: 'event', id, kind, direction: ev.direction, level: ev.level, atSec });
  }

  return (
    <div className="card">
      <div className="card-h">
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M3 17l6-6 4 4 8-8" />
          <path d="M21 3v6h-6" />
        </svg>
        <h3>{tr('regime.title')}</h3>
        <span className="hsp" />
        <button
          type="button"
          className={cn('hbtn', isRegime && 'on')}
          aria-label={t('help')}
          aria-expanded={isRegime}
          onClick={() => toggle('regime', 'concept')}
        >
          ?
        </button>
      </div>

      <div className="tgrid">
        {shown.map((tile, i) => {
          const isOpen = open?.key === tile.key;
          const isLastOdd = shown.length % 2 === 1 && i === shown.length - 1;
          return (
            <button
              type="button"
              key={tile.key}
              className={cn('tile', isOpen && 'on', isLastOdd && 'span2')}
              aria-expanded={isOpen}
              onClick={() => toggle(tile.key, 'data')}
            >
              <div className="k">
                {tile.label}
                <span
                  role="button"
                  tabIndex={0}
                  className={cn('thelp', isOpen && activeTab === 'concept' && 'on')}
                  aria-label={t('measureHelp')}
                  onClick={(e) => {
                    e.stopPropagation();
                    toggle(tile.key, 'concept');
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      e.stopPropagation();
                      toggle(tile.key, 'concept');
                    }
                  }}
                >
                  ?
                </span>
              </div>
              {tile.html ? (
                tile.html
              ) : (
                <div className={cn('v', tile.mono && 'mono', tile.value == null && 'na')}>
                  {tile.value ?? t('unavailable')}
                </div>
              )}
              {tile.sub && <div className="sub">{tile.sub}</div>}
              <svg className="tarrow" viewBox="0 0 24 24" aria-hidden>
                <path d="M4 12h15M13 6l6 6-6 6" />
              </svg>
            </button>
          );
        })}
      </div>

      {open && (
        <div className="tdetail">
          <div className="tdhead">
            <span className="tdt">{panelTitle}</span>
            <span className="tdsp" />
            <div className="ttabs" role="tablist">
              {(isRegime ? (['concept'] as Tab[]) : (['data', 'concept'] as Tab[])).map((tab) => (
                <button
                  key={tab}
                  type="button"
                  role="tab"
                  aria-selected={activeTab === tab}
                  className={cn('ttab', activeTab === tab && 'on')}
                  onClick={() => toggle(open.key, tab)}
                >
                  {tab === 'data' ? t('tabData') : t('tabConcept')}
                </button>
              ))}
            </div>
            <button
              type="button"
              className="tdclose"
              aria-label={t('close')}
              onClick={() => toggle(open.key, open.tab)}
            >
              ×
            </button>
          </div>
          <div className="tdbody">
            {activeTab === 'concept' ? (
              <Concept k={open.key} />
            ) : (
              renderData(open.key, {
                t,
                fmt,
                px,
                dayHm,
                tf,
                regime,
                structure,
                header,
                price,
                priceTimeLabel,
                range,
                pos,
                vd,
                avail,
                order: mtfOrder,
                trends,
                domKind,
                maturity,
                last,
                lastKind,
                obActive,
                obTested,
                obMitigated,
                fvgActive,
                fvgPartial,
                fvgOpen,
                session,
                weeklyClose: formatWeeklyClose(marketStatus?.weekly_close, locale),
                sessionName,
                transitionName,
                fmtDelay,
                levelRows,
                levelsInsufficient,
                referenceLevel,
                traceLevel,
                selectEvent,
                selectedEventId,
                refLabel,
                sideOf: coincidentSide,
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Concept tab (static, versioned i18n + mandatory « ne dit pas ») ──────────
function Concept({ k }: { k: string }) {
  const t = useTranslations('regimePanel');
  const title = t(`concept.${k}.title`);
  const body = t(`concept.${k}.body`);
  const notSay = t(`concept.${k}.notSay`);
  const paragraphs = body.split(/\n\n+/).filter(Boolean);
  return (
    <>
      <Dh4>{title}</Dh4>
      {paragraphs.map((p, i) => (
        <Dp key={i}>{withBold(p)}</Dp>
      ))}
      <div className="notsay">
        <div className="nt">{t('notSayLabel')}</div>
        <p>{withBold(notSay)}</p>
      </div>
    </>
  );
}

// ─── Donnée tab (live engine values — the proof) ──────────────────────────────
interface DataCtx {
  t: ReturnType<typeof useTranslations>;
  fmt: ReturnType<typeof useReadingFormatters>;
  px: (v: number) => string;
  dayHm: (iso: string) => string;
  tf: string;
  regime: MarketReadingRegime;
  structure: MarketReadingStructure;
  header: MarketReadingHeader;
  price: number;
  /** Preformatted « Prix à HH:MM » label for the current-price row, or null. */
  priceTimeLabel: string | null;
  range: { low: number; high: number } | null;
  pos: number | null;
  vd: MarketReadingRegime['volatility_detail'];
  avail: { key: string; label: string }[];
  order: { key: string; label: string; tf: string }[];
  trends: Record<string, 'bullish' | 'bearish' | 'indeterminate' | null>;
  domKind: 'up' | 'down' | 'flat';
  maturity: ReturnType<typeof deriveTrendMaturity>;
  last: (BOSRecent | CHOCHRecent) | null;
  lastKind: 'CHOCH' | 'BOS' | null;
  obActive: unknown[];
  obTested: unknown[];
  obMitigated: unknown[];
  fvgActive: unknown[];
  fvgPartial: unknown[];
  fvgOpen: number;
  session: SessionInfo | null;
  weeklyClose: string | null;
  sessionName: (name: SessionName) => string;
  transitionName: (label: TransitionLabel) => string;
  fmtDelay: (mins: number) => string;
  levelRows: { key: string; label: string; value: number }[];
  levelsInsufficient: string[];
  referenceLevel: { price: number; label: string } | null;
  traceLevel: (label: string, value: number) => void;
  /** VZ-1b — select a BOS/CHOCH event (temporal frame on its confirmation bar). */
  selectEvent: (
    kind: 'bos' | 'choch',
    ev: { broken_at: string; direction: 'bullish' | 'bearish'; level: number },
  ) => void;
  selectedEventId: string | null;
  refLabel: (label: string, value: number) => string;
  /** Side of a detected pocket coinciding with a price level, or null (RG-1d). */
  sideOf: (value: number) => LiquiditySide | null;
}

/**
 * A price gap expressed in WORDS, never a bare sign (RG-1d): « 1,33 % au-dessus »
 * / « 1,23 % en dessous » / « au prix courant » when the gap rounds to 0,00 % at
 * the displayed precision. The word carries the direction — no « + » / « − ».
 * `pct` is (level − price) / price × 100 (positive ⇒ level above the price).
 */
function relWord(pct: number | null, t: (k: string, v?: Record<string, string>) => string): string {
  if (pct == null) return '';
  const abs = Math.abs(pct);
  if (abs < 0.005) return t('data.relAt'); // rounds to 0,00 % → « au prix courant »
  const val = abs.toFixed(2);
  return pct > 0 ? t('data.relAbove', { pct: val }) : t('data.relBelow', { pct: val });
}

function renderData(k: string, c: DataCtx): React.ReactNode {
  const { t, fmt, px, dayHm, tf } = c;
  switch (k) {
    case 'phase':
      return (
        <>
          <Dh4>{t('data.phaseHead')}</Dh4>
          <div className="ev">
            <EvRow k={t('data.trendRow')} v={fmt.trend(c.regime.trend).label} />
            <EvRow k={t('data.volRow')} v={fmt.volatility(c.regime.volatility_observed).label} />
            <EvRow k={t('data.ruleRow')} v={fmt.marketPhaseShort(c.regime.market_phase)} />
          </div>
          <Dp>{c.t('data.phaseNote')}</Dp>
        </>
      );
    case 'trend': {
      // TR-1: the trend is STRUCTURAL — the direction of the last non-contradicted
      // BOS/CHOCH — not a close-delta over a fixed window. Name the anchoring
      // event so the client can trace WHY it reads as it does. (The MT-D1
      // per-unit analysis window no longer describes the trend — it bounds the
      // event journal, surfaced on the Structure card, not here.)
      const ref = c.regime.trend_reference ?? null;
      return (
        <>
          <Dh4>{t('data.trendHead')}</Dh4>
          <div className="ev">
            <EvRow k={t('data.resultRow')} v={fmt.trend(c.regime.trend).label} />
            <EvRow k={t('data.measuredOnRow')} v={tf} />
            {ref ? (
              <EvRow
                k={t('data.trendAnchorRow')}
                v={t('data.trendAnchorValue', {
                  kind: ref.kind === 'choch' ? t('value.kindChoch') : t('value.kindBos'),
                  dir: ref.direction === 'bullish' ? t('value.dirUp') : t('value.dirDown'),
                })}
                t={dayHm(ref.broken_at)}
              />
            ) : (
              <EvRow k={t('data.trendAnchorRow')} v={t('data.trendAnchorNone')} />
            )}
          </div>
          <Dp>{t('data.trendNoteStructural')}</Dp>
        </>
      );
    }
    case 'vol': {
      const vd = c.vd;
      if (!vd) return <Dp>{t('data.trendNote', { tf })}</Dp>;
      return (
        <>
          <Dh4>{t('data.volHead')}</Dh4>
          <div className="ev">
            <EvRow k={t('data.volRecent', { n: vd.recent_n })} v={c.px(vd.recent_avg)} />
            <EvRow k={t('data.volBaseline', { n: vd.baseline_n })} v={c.px(vd.baseline_avg)} />
            <EvRow k={t('data.volRatio')} v={vd.ratio.toFixed(2)} />
            <EvRow
              k={t('data.volThresholds')}
              v={t('data.volThresholdsValue', {
                low: vd.threshold_low.toFixed(2),
                high: vd.threshold_high.toFixed(2),
              })}
            />
          </div>
          <Dp>{t('data.volNote', { n: vd.baseline_n })}</Dp>
        </>
      );
    }
    case 'pos': {
      if (!c.range || c.pos == null) return null;
      const distTop = distancePct(c.range.high, c.price);
      const distBottom = distancePct(c.range.low, c.price);
      return (
        <>
          <Dh4>{t('data.posHead')}</Dh4>
          <div className="bararea">
            <div className="barwrap">
              <i style={{ width: `${c.pos}%` }} />
              <u style={{ left: `${c.pos}%` }} />
            </div>
            <div className="barlbl">
              <PxBtn label={t('data.boundLow')} value={c.range.low} c={c}>
                {t('data.barLow', { v: px(c.range.low) })}
              </PxBtn>
              <PxBtn label={t('data.boundHigh')} value={c.range.high} c={c}>
                {t('data.barHigh', { v: px(c.range.high) })}
              </PxBtn>
            </div>
          </div>
          <div className="ev">
            <EvRow k={t('data.currentPrice')} v={px(c.price)} t={c.priceTimeLabel ?? undefined} />
            <EvRow k={t('data.distTop')} v={px(c.range.high - c.price)} t={relWord(distTop, t)} />
            <EvRow k={t('data.distBottom')} v={px(c.price - c.range.low)} t={relWord(distBottom, t)} />
            <EvRow k={t('data.rangeSpan')} v={px(c.range.high - c.range.low)} />
          </div>
          <Dp>{withBold(t('data.posNote'))}</Dp>
        </>
      );
    }
    case 'align':
      return (
        <>
          <Dh4>{t('data.alignHead')}</Dh4>
          <div className="ev">
            {c.order.length === 0 ? (
              <div className="tfline"><span className="st">{c.t('value.alignNone')}</span></div>
            ) : (
              c.order.map(({ key, label }) => {
                const dir = c.trends[key];
                const kind = dir === 'bullish' ? 'up' : dir === 'bearish' ? 'down' : 'flat';
                const diverges = c.domKind !== 'flat' && kind !== 'flat' && kind !== c.domKind;
                // An unavailable upper unit reads as indisponible, never an agreement.
                const label2 = dir == null ? c.t('data.alignUnavailable') : fmt.trend(dir).label;
                return (
                  <div className={cn('tfline', diverges && 'dis')} key={key}>
                    <span className="tf">{label}</span>
                    <span className="st">{label2}</span>
                    <span className="ev-t">{arrowOf(dir ?? null)}</span>
                  </div>
                );
              })
            )}
          </div>
          <Dp>{t('data.alignNote')}</Dp>
        </>
      );
    case 'mat': {
      const m = c.maturity;
      if (!m) return null;
      const anchor = latestBreak(c.structure.choch_events, c.structure.current_choch);
      const closeMs = new Date(c.header.candle_close_ts).getTime();
      const breakMs = new Date(m.brokenAt).getTime();
      const mins = Math.max(0, Math.floor((closeMs - breakMs) / 60000));
      const elapsed = `${Math.floor(mins / 60)} h ${String(mins % 60).padStart(2, '0')}`;
      const since = [
        ...(c.structure.bos_events ?? []).map((e) => ({ kind: 'bos' as const, e })),
        ...(c.structure.choch_events ?? []).map((e) => ({ kind: 'choch' as const, e })),
      ]
        .filter(({ e }) => new Date(e.broken_at).getTime() > breakMs)
        .sort((a, b) => new Date(b.e.broken_at).getTime() - new Date(a.e.broken_at).getTime());
      return (
        <>
          <Dh4>{t('data.matHead')}</Dh4>
          <div className="ev">
            <EvRow
              k={t('data.anchorEvent')}
              v={`CHOCH ${m.direction === 'bullish' ? '↑' : '↓'}`}
              t={dayHm(m.brokenAt)}
            />
            {anchor && (
              <EvRow
                k={t('data.crossedLevel')}
                v={<EvBtn kind="choch" ev={anchor} c={c} />}
              />
            )}
            <EvRow k={t('data.barsSince')} v={String(m.bars ?? '—')} t={tf} />
            <EvRow k={t('data.elapsed')} v={elapsed} />
          </div>
          {since.length > 0 && (
            <>
              <Dh4 mt>{t('data.eventsSince')}</Dh4>
              <div className="ev">
                {since.map(({ kind, e }, i) => {
                  const kLabel = `${kind.toUpperCase()} ${e.direction === 'bullish' ? '↑' : '↓'}`;
                  return (
                    <EvRow
                      key={i}
                      k={kLabel}
                      v={<EvBtn kind={kind} ev={e} c={c} />}
                      t={dayHm(e.broken_at)}
                    />
                  );
                })}
              </div>
            </>
          )}
        </>
      );
    }
    case 'last': {
      const l = c.last;
      if (!l) return null;
      const journal = [
        ...(c.structure.bos_events ?? []).map((e) => ({ kind: 'BOS' as const, e })),
        ...(c.structure.choch_events ?? []).map((e) => ({ kind: 'CHOCH' as const, e })),
      ].sort((a, b) => new Date(b.e.broken_at).getTime() - new Date(a.e.broken_at).getTime());
      return (
        <>
          <Dh4>{t('data.lastHead')}</Dh4>
          <div className="ev">
            <EvRow
              k={t('data.typeRow')}
              v={`${c.lastKind} ${l.direction === 'bullish' ? '↑' : '↓'}`}
            />
            <EvRow
              k={t('data.crossedExtreme')}
              v={<EvBtn kind={c.lastKind === 'CHOCH' ? 'choch' : 'bos'} ev={l} c={c} />}
              t={dayHm(l.broken_at)}
            />
          </div>
          {journal.length > 0 && (
            <>
              <Dh4 mt>{t('data.journal', { tf })}</Dh4>
              {(() => {
                // MT-D1 fix: the journal shows EVERY break in the analysis window
                // (no more silent slice(0,6)) and LABELS the window so the count is
                // never mistaken for « all history ». Window size + span are the
                // REAL per-unit analysed bars from the header (fallback 500).
                const windowBars = c.header.analysis_window_bars ?? TREND_WINDOW_BARS;
                const span = trendWindowSpan(tf, windowBars);
                const spanText = span ? t(`data.windowSpan_${span.unit}`, { count: span.count }) : '';
                return (
                  <Dp>{t('data.journalCount', {
                    count: journal.length,
                    bars: windowBars,
                    span: spanText,
                  })}</Dp>
                );
              })()}
              <div className="ev">
                {journal.map(({ kind, e }, i) => {
                  const kLabel = `${kind} ${e.direction === 'bullish' ? '↑' : '↓'}`;
                  return (
                    <EvRow
                      key={i}
                      k={kLabel}
                      v={<EvBtn kind={kind === 'CHOCH' ? 'choch' : 'bos'} ev={e} c={c} />}
                      t={dayHm(e.broken_at)}
                    />
                  );
                })}
              </div>
            </>
          )}
        </>
      );
    }
    case 'dens':
      return (
        <>
          <Dh4>{t('data.densHead', { tf })}</Dh4>
          <div className="ev">
            <EvRow k={t('data.obActive')} v={String(c.obActive.length)} />
            <EvRow k={t('data.obTested')} v={String(c.obTested.length)} />
            <EvRow k={t('data.obMitigated')} v={String(c.obMitigated.length)} />
            <EvRow k={t('data.fvgOpen')} v={String(c.fvgOpen)} />
            <EvRow k={t('data.fvgUntouched')} v={String(c.fvgActive.length)} />
            <EvRow k={t('data.fvgPartial')} v={String(c.fvgPartial.length)} />
          </div>
          <Dp>{withBold(t('data.densNote'))}</Dp>
        </>
      );
    case 'sess': {
      const s = c.session;
      if (!s) return null;
      // Continuous market (crypto): no découpage, no weekly close.
      if (s.current === 'continuous') {
        return (
          <>
            <Dh4>{t('data.sessHead')}</Dh4>
            <Dp>{c.sessionName('continuous')}</Dp>
          </>
        );
      }
      const ov = s.overlap;
      const overlapState = ov ? t(`data.overlap.${ov.state}`) : null;
      return (
        <>
          <Dh4>{t('data.sessHead')}</Dh4>
          <div className="ev">
            <EvRow
              k={t('data.sessCurrent')}
              v={c.sessionName(s.current)}
              t={s.currentRange ? `${s.currentRange.start} – ${s.currentRange.end}` : undefined}
            />
            {s.localTime && <EvRow k={t('data.marketLocalTime')} v={s.localTime} />}
            {s.next && (
              <EvRow
                k={t('data.sessNext')}
                v={`${c.transitionName(s.next.label)} ${s.next.at}`}
                t={c.fmtDelay(s.next.inMinutes)}
              />
            )}
            {ov && (
              <EvRow k={t('data.sessOverlap')} v={`${ov.start} – ${ov.end}`} t={overlapState ?? undefined} />
            )}
            {c.weeklyClose && <EvRow k={t('data.weeklyClose')} v={c.weeklyClose} />}
          </div>
        </>
      );
    }
    case 'lvl': {
      if (c.levelRows.length === 0) return null;
      return (
        <>
          <Dh4>{t('data.lvlHead', { price: px(c.price) })}</Dh4>
          <div className="ev">
            {c.levelRows.map((r) => {
              // A small factual mark BEFORE clicking when the level coincides with
              // a detected pocket — « ≈ BSL » — never a ranking or a target (RG-1d).
              const side = c.sideOf(r.value);
              return (
                <div className="ev-r" key={r.key}>
                  <span className="ev-k">
                    {r.label}
                    {side && (
                      <span className="coin" title={t('data.coincidence')}>
                        {t('data.coincidenceMark', { side: formatLiquiditySideShort(side) })}
                      </span>
                    )}
                  </span>
                  <span className="ev-t">{relWord(distancePct(r.value, c.price), t)}</span>
                  <PxBtn label={r.label} value={r.value} c={c} />
                </div>
              );
            })}
          </div>
          {c.levelsInsufficient.length > 0 && (
            <Dp>{t('data.lvlInsufficient', { periods: c.levelsInsufficient.join(', ') })}</Dp>
          )}
          <Dp>{t('data.lvlNote')}</Dp>
        </>
      );
    }
    default:
      return null;
  }
}
