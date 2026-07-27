'use client';

import * as React from 'react';
import { useLocale, useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import { useMtfTrends } from '@/lib/market-reading/hooks';
import { MTF_TREND_ORDER } from '@/lib/market-reading/mtf-trend';
import { deriveTrendMaturity } from '@/lib/market-reading/regime-facts';
import { formatBreakTimestamp } from '@/lib/market-reading/regime-facts';
import { fetchCandles } from '@/lib/market-reading/api-client';
import {
  structureRange,
  positionPct,
  referenceLevels,
  distancePct,
  type ReferenceLevels,
} from '@/lib/market-reading/reference-levels';
import { useChartViewOptional } from '@/lib/chart/viewState';
import type { MarketStatusView } from '@/lib/market-reading/status';
import type {
  BOSRecent,
  CHOCHRecent,
  Candle,
  MarketReadingHeader,
  MarketReadingRegime,
  MarketReadingStructure,
} from '@/types/market-reading';
import './regime-tiles.css';

// ─── openHelp coordination (one panel open across the whole page) ─────────────
// The regime detail panel rides the page-wide `openHelp` string so opening it
// closes any other card's help and vice-versa. Encoding: `rg:<tileKey>:<tab>`.
const RG = 'rg:';
type Tab = 'data' | 'concept';
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

function arrowOf(dir: 'bullish' | 'bearish' | 'neutral' | 'ranging' | null): string {
  return dir === 'bullish' ? '↑' : dir === 'bearish' ? '↓' : '→';
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

/** Fetch the D1 / W1 reference candle series once per instrument (read-only). */
function useReferenceCandles(instrument: string): { daily: Candle[]; weekly: Candle[] } | null {
  const [state, setState] = React.useState<{ daily: Candle[]; weekly: Candle[] } | null>(null);
  React.useEffect(() => {
    let alive = true;
    setState(null);
    const controller = new AbortController();
    Promise.all([
      fetchCandles(instrument, 'D1', { limit: 40, signal: controller.signal }).catch(() => [] as Candle[]),
      fetchCandles(instrument, 'W1', { limit: 20, signal: controller.signal }).catch(() => [] as Candle[]),
    ]).then(([daily, weekly]) => {
      if (alive) setState({ daily, weekly });
    });
    return () => {
      alive = false;
      controller.abort();
    };
  }, [instrument]);
  return state;
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

interface RegimeCardProps {
  regime: MarketReadingRegime;
  structure: MarketReadingStructure;
  header: MarketReadingHeader;
  price: number;
  marketStatus: MarketStatusView | null;
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
  marketStatus,
  openHelp,
  onToggleHelp,
}: RegimeCardProps) {
  const t = useTranslations('regimePanel');
  const tr = useTranslations('reading');
  const locale = useLocale();
  const fmt = useReadingFormatters();
  const instrument = header.instrument;
  const tf = header.timeframe;
  const { trends } = useMtfTrends(instrument);
  const { referenceLevel, setReferenceLevel } = useChartViewOptional();

  const refCandles = useReferenceCandles(instrument);
  const refLevels: ReferenceLevels | null = refCandles
    ? referenceLevels(refCandles.daily, refCandles.weekly)
    : null;

  const px = (v: number) => fmt.price(v, instrument);
  const dayHm = (iso: string) => formatBreakTimestamp(iso) ?? '';

  // ── Derived facts (all read-only over engine output) ────────────────────────
  const maturity = deriveTrendMaturity(structure, header);

  const avail = MTF_TREND_ORDER.filter(({ key }) => trends[key] != null);
  const dirs = avail.map(({ key }) =>
    trends[key] === 'bullish' ? 'up' : trends[key] === 'bearish' ? 'down' : 'flat',
  );
  const up = dirs.filter((d) => d === 'up').length;
  const down = dirs.filter((d) => d === 'down').length;
  const flat = dirs.filter((d) => d === 'flat').length;
  let domArrow = '→';
  let domKind: 'up' | 'down' | 'flat' = 'flat';
  let aligned = flat;
  if (up >= down && up >= flat) {
    domArrow = '↑';
    domKind = 'up';
    aligned = up;
  } else if (down >= up && down >= flat) {
    domArrow = '↓';
    domKind = 'down';
    aligned = down;
  }

  const lc = latestBreak(structure.choch_events, structure.choch);
  const lb = latestBreak(structure.bos_events, structure.bos);
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

  // Market hours (re-scoped Session → MC-1). New York wall-clock snapshot.
  const nyTime = React.useMemo(() => {
    try {
      return new Intl.DateTimeFormat(locale, {
        timeZone: 'America/New_York',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      }).format(new Date());
    } catch {
      return null;
    }
  }, [locale, price]); // re-snapshot on tick

  // Reference-level rows (measure 10), each present only if computable.
  const levelRows: { key: keyof ReferenceLevels; label: string; value: number }[] = [];
  if (refLevels) {
    const order: [keyof ReferenceLevels, string][] = [
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

  // ── Tile facades ────────────────────────────────────────────────────────────
  const tiles: Tile[] = [
    {
      key: 'phase',
      label: t('tiles.phase'),
      value: fmt.marketPhaseShort(regime.market_phase),
      sub: t('sub.phase'),
      available: true,
    },
    {
      key: 'trend',
      label: t('tiles.trend'),
      value: fmt.trend(regime.trend).label,
      sub: t('sub.trend', { tf }),
      available: true,
    },
    {
      key: 'vol',
      label: t('tiles.vol'),
      value: fmt.volatility(regime.volatility_observed).label,
      sub: t('sub.vol'),
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
        avail.length > 0
          ? t('value.align', { count: aligned, total: avail.length, arrow: domArrow })
          : null,
      mono: true,
      sub:
        avail.length > 0
          ? avail.map(({ key, label }) => `${label} ${arrowOf(trends[key])}`).join(' · ')
          : null,
      available: avail.length > 0,
    },
    {
      key: 'mat',
      label: t('tiles.mat'),
      value: maturity?.bars != null ? t('value.mat', { count: maturity.bars }) : null,
      mono: true,
      sub: maturity ? t('sub.mat', { date: dayHm(maturity.brokenAt) }) : null,
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
      label: t('tiles.sess'),
      value: marketStatus ? t(`state.${marketStatus.state}`) : null,
      sub: marketStatus
        ? marketStatus.isClosed && marketStatus.nextOpenTs
          ? `${t('data.nextOpen')} · ${dayHm(marketStatus.nextOpenTs)}`
          : nyTime
            ? `${t('data.marketLocalTime')} · ${nyTime}`
            : t('data.weeklyCloseValue')
        : null,
      available: marketStatus != null,
    },
    {
      key: 'lvl',
      label: t('tiles.lvl'),
      html:
        levelRows.length > 0 ? (
          <>
            {levelRows.slice(2, 4).map((r) => (
              <div className="lvlmini" key={r.key}>
                <span>{r.label}</span>
                <b>{px(r.value)}</b>
              </div>
            ))}
          </>
        ) : null,
      sub: levelRows.length > 0 ? t('sub.lvl', { count: levelRows.length }) : null,
      available: levelRows.length > 0,
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

  function traceLevel(labelKey: string, label: string, value: number) {
    const same = referenceLevel != null && referenceLevel.price === value && referenceLevel.label === label;
    setReferenceLevel(same ? null : { price: value, label });
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
                range,
                pos,
                vd,
                avail,
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
                marketStatus,
                nyTime,
                levelRows,
                referenceLevel,
                traceLevel,
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
  range: { low: number; high: number } | null;
  pos: number | null;
  vd: MarketReadingRegime['volatility_detail'];
  avail: { key: string; label: string }[];
  trends: Record<string, 'bullish' | 'bearish' | 'neutral' | 'ranging' | null>;
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
  marketStatus: MarketStatusView | null;
  nyTime: string | null;
  levelRows: { key: string; label: string; value: number }[];
  referenceLevel: { price: number; label: string } | null;
  traceLevel: (labelKey: string, label: string, value: number) => void;
}

function fmtSigned(pct: number | null): string {
  if (pct == null) return '';
  const sign = pct > 0 ? '+ ' : pct < 0 ? '− ' : '';
  return `${sign}${Math.abs(pct).toFixed(2)} %`;
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
    case 'trend':
      return (
        <>
          <Dh4>{t('data.trendHead')}</Dh4>
          <div className="ev">
            <EvRow k={t('data.resultRow')} v={fmt.trend(c.regime.trend).label} />
            <EvRow k={t('data.measuredOnRow')} v={tf} />
          </div>
          <Dp>{t('data.trendNote', { tf })}</Dp>
        </>
      );
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
              <span>{t('data.barLow', { v: px(c.range.low) })}</span>
              <span>{t('data.barHigh', { v: px(c.range.high) })}</span>
            </div>
          </div>
          <div className="ev">
            <EvRow k={t('data.currentPrice')} v={px(c.price)} />
            <EvRow k={t('data.distTop')} v={px(c.range.high - c.price)} t={fmtSigned(distTop)} />
            <EvRow k={t('data.distBottom')} v={px(c.price - c.range.low)} t={fmtSigned(distBottom)} />
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
            {c.avail.map(({ key, label }) => {
              const dir = c.trends[key];
              const kind = dir === 'bullish' ? 'up' : dir === 'bearish' ? 'down' : 'flat';
              const diverges = c.domKind !== 'flat' && kind !== 'flat' && kind !== c.domKind;
              return (
                <div className={cn('tfline', diverges && 'dis')} key={key}>
                  <span className="tf">{label}</span>
                  <span className="st">{fmt.trend(dir ?? 'neutral').label}</span>
                  <span className="ev-t">{arrowOf(dir ?? null)}</span>
                </div>
              );
            })}
          </div>
          <Dp>{t('data.alignNote')}</Dp>
        </>
      );
    case 'mat': {
      const m = c.maturity;
      if (!m) return null;
      const anchor = latestBreak(c.structure.choch_events, c.structure.choch);
      const closeMs = new Date(c.header.candle_close_ts).getTime();
      const breakMs = new Date(m.brokenAt).getTime();
      const mins = Math.max(0, Math.floor((closeMs - breakMs) / 60000));
      const elapsed = `${Math.floor(mins / 60)} h ${String(mins % 60).padStart(2, '0')}`;
      const since = [...(c.structure.bos_events ?? []), ...(c.structure.choch_events ?? [])]
        .filter((e) => new Date(e.broken_at).getTime() > breakMs)
        .sort((a, b) => new Date(b.broken_at).getTime() - new Date(a.broken_at).getTime());
      return (
        <>
          <Dh4>{t('data.matHead')}</Dh4>
          <div className="ev">
            <EvRow
              k={t('data.anchorEvent')}
              v={`CHOCH ${m.direction === 'bullish' ? '↑' : '↓'}`}
              t={dayHm(m.brokenAt)}
            />
            {anchor && <EvRow k={t('data.crossedLevel')} v={px(anchor.level)} />}
            <EvRow k={t('data.barsSince')} v={String(m.bars ?? '—')} t={tf} />
            <EvRow k={t('data.elapsed')} v={elapsed} />
          </div>
          {since.length > 0 && (
            <>
              <Dh4 mt>{t('data.eventsSince')}</Dh4>
              <div className="ev">
                {since.slice(0, 6).map((e, i) => (
                  <EvRow
                    key={i}
                    k={`${'validation_status' in e && e === anchor ? 'CHOCH' : 'BOS'} ${e.direction === 'bullish' ? '↑' : '↓'}`}
                    v={px(e.level)}
                    t={dayHm(e.broken_at)}
                  />
                ))}
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
            <EvRow k={t('data.crossedExtreme')} v={px(l.level)} t={dayHm(l.broken_at)} />
          </div>
          {journal.length > 0 && (
            <>
              <Dh4 mt>{t('data.journal', { tf })}</Dh4>
              <div className="ev">
                {journal.slice(0, 6).map(({ kind, e }, i) => (
                  <EvRow
                    key={i}
                    k={`${kind} ${e.direction === 'bullish' ? '↑' : '↓'}`}
                    v={px(e.level)}
                    t={dayHm(e.broken_at)}
                  />
                ))}
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
      const s = c.marketStatus;
      if (!s) return null;
      return (
        <>
          <Dh4>{t('data.sessHead')}</Dh4>
          <div className="ev">
            <EvRow k={t('data.marketState')} v={t(`state.${s.state}`)} />
            {c.nyTime && <EvRow k={t('data.marketLocalTime')} v={c.nyTime} />}
            <EvRow k={t('data.weeklyClose')} v={t('data.weeklyCloseValue')} />
            {s.isClosed && s.nextOpenTs && (
              <EvRow k={t('data.nextOpen')} v={dayHm(s.nextOpenTs)} />
            )}
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
              const on =
                c.referenceLevel != null &&
                c.referenceLevel.price === r.value &&
                c.referenceLevel.label === r.label;
              return (
                <div className="ev-r" key={r.key}>
                  <span className="ev-k">{r.label}</span>
                  <span className="ev-t">{fmtSigned(distancePct(r.value, c.price))}</span>
                  <button
                    type="button"
                    className={cn('pxbtn', on && 'on')}
                    aria-pressed={on}
                    onClick={() => c.traceLevel(r.key, r.label, r.value)}
                  >
                    {px(r.value)}
                  </button>
                </div>
              );
            })}
          </div>
          <Dp>{t('data.lvlNote')}</Dp>
        </>
      );
    }
    default:
      return null;
  }
}
