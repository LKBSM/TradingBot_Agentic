'use client';

import * as React from 'react';
import dynamic from 'next/dynamic';
import { Loader2 } from 'lucide-react';
import { useTranslations } from 'next-intl';
import {
  ChartUnavailable,
  EmptyReadingState,
  ReadingErrorState,
} from './ReadingPlaceholders';
import { ReadingSkeleton } from './ReadingSkeleton';
import { READING_DATA_SOURCE } from '@/lib/mockReadings';
import {
  useCandles,
  useLatestPrice,
  useMtfTrends,
  type ReadingSource,
} from '@/lib/market-reading/hooks';
import { useLivePrice } from '@/lib/market-reading/live-price';
import { useMarketClosed } from '@/lib/market-reading/session';
import { useChartViewOptional } from '@/lib/chart/viewState';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import {
  MTF_TREND_ORDER,
  classifyMtfAlignment,
  mtfTrendGlyph,
} from '@/lib/market-reading/mtf-trend';
import { deriveTrendMaturity } from '@/lib/market-reading/regime-facts';
import type { Combo } from '@/lib/market-reading/store';
import type {
  Candle,
  MarketReading,
  MarketReadingConditions,
  MarketReadingEvents,
  MarketReadingHeader,
  MarketReadingRegime,
  MarketReadingStructure,
} from '@/types/market-reading';

/**
 * Desktop /app reading — the terminal, flat-grid reproduction of
 * docs/design/reference-desktop.html (mission UI-2). It renders the SAME data
 * the mobile ReadingColumn shows (nothing is invented) in the reference shell
 * markup: apphead · legalbar · chart pad with layer pills · a two-column panel
 * grid. Every value comes from the typed reading via useReadingFormatters — the
 * detection engine and its API are untouched.
 *
 * Desktop-only: the mobile/tablet layout (<1280px) keeps ReadingColumn +
 * MarketReadingCard (the accordion), which is the right shape for narrow widths.
 */
const ReadingChart = dynamic(
  () => import('./ReadingChart').then((m) => ({ default: m.ReadingChart })),
  {
    ssr: false,
    loading: () => (
      <div
        className="flex h-[clamp(300px,52svh,560px)] w-full items-center justify-center rounded-md border border-border/60 bg-muted/30"
        role="status"
      >
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" aria-hidden />
      </div>
    ),
  },
);

interface DesktopReadingProps {
  active: Combo | null;
  reading: MarketReading | null;
  isLoading: boolean;
  isRefreshing: boolean;
  error: Error | null;
  onRetry: () => void;
  dataSource?: ReadingSource;
}

export function DesktopReading({
  active,
  reading,
  isLoading,
  isRefreshing,
  error,
  onRetry,
  dataSource = READING_DATA_SOURCE,
}: DesktopReadingProps) {
  const t = useTranslations('app');

  // Candle + price feeds — identical wiring to ReadingColumn (mobile), so the
  // desktop layout reads exactly the same honest, candle-close data.
  const { candles } = useCandles(
    active?.instrument ?? null,
    active?.timeframe ?? null,
    { source: dataSource, candleCloseTs: reading?.header.candle_close_ts ?? null },
  );
  const { change: live } = useLatestPrice(active?.instrument ?? null, {
    source: dataSource,
    candleCloseTs: reading?.header.candle_close_ts ?? null,
  });
  const { price: livePrice, ts: liveTs } = useLivePrice(active?.instrument ?? null, {
    enabled: dataSource === 'live' ? undefined : false,
  });

  const { view: chartView, applyActions } = useChartViewOptional();

  const onClearHighlight = React.useCallback(() => {
    applyActions([{ action: 'clear_highlight', params: {} }]);
  }, [applyActions]);

  const liveHeader = React.useMemo(() => {
    if (live == null || livePrice == null || !Number.isFinite(livePrice)) return live;
    const ref = live.referenceClose;
    return {
      ...live,
      price: livePrice,
      priceTs: liveTs ?? live.priceTs,
      changeAbs: ref != null ? livePrice - ref : live.changeAbs,
      changePct: ref != null && ref !== 0 ? (livePrice - ref) / ref : live.changePct,
    };
  }, [live, livePrice, liveTs]);

  const marketClosed = useMarketClosed(
    active?.instrument ?? null,
    liveHeader?.priceTs ?? null,
  );

  if (!active) return <EmptyReadingState />;
  if (error) return <ReadingErrorState error={error} onRetry={onRetry} />;
  if (isLoading && !reading) return <ReadingSkeleton />;
  if (!reading) return <ReadingSkeleton />;

  const header = reading.header;
  const price = liveHeader?.price ?? header.close_price;
  const changeAbs = liveHeader?.changeAbs ?? null;

  const chartSlot =
    candles && candles.length > 0 ? (
      <ReadingChart
        candles={candles}
        structure={reading.structure}
        instrument={header.instrument}
        timeframe={active.timeframe}
        livePrice={livePrice}
        liveTs={liveTs}
        marketClosed={marketClosed}
        layers={chartView.layers}
        filter={chartView.filter}
        focus={chartView.focus}
        highlightZoneId={chartView.highlightZoneId}
        onClearHighlight={onClearHighlight}
        hiddenZoneIds={chartView.hiddenZoneIds}
        isolatedZoneIds={chartView.isolatedZoneIds}
      />
    ) : (
      <ChartUnavailable />
    );

  return (
    <div className="cv-app" aria-label={t('column.sectionAria')}>
      {isRefreshing && (
        <div
          className="flex items-center gap-2 px-[18px] pt-3 text-xs text-muted-foreground"
          role="status"
          aria-live="polite"
        >
          <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
          {t('column.refreshing')}
        </div>
      )}

      <AppHead
        instrument={header.instrument}
        price={price}
        changeAbs={changeAbs}
        marketClosed={marketClosed}
      />

      <LegalBar />

      <div className="chartpad">
        <LayerPills
          layers={chartView.layers}
          activeOnly={chartView.filter.activeOnly}
          onToggleLayer={(layer, on) =>
            applyActions([{ action: 'set_layer_visibility', params: { layer, visible: !on } }])
          }
          onToggleMitigated={(shown) =>
            applyActions([{ action: 'filter_zones', params: { active_only: shown } }])
          }
        />
        <div className="chartbox">{chartSlot}</div>
      </div>

      <div className="panels">
        <NarratedPanel conditions={reading.conditions} />
        <RegimePanel regime={reading.regime} structure={reading.structure} header={header} />
        <StructurePanel structure={reading.structure} instrument={header.instrument} />
        <LiquidityPanel structure={reading.structure} instrument={header.instrument} />
        <NewsPanel events={reading.events} />
      </div>
    </div>
  );
}

/* ── Head + legal bar ─────────────────────────────────────────────────────── */

function AppHead({
  instrument,
  price,
  changeAbs,
  marketClosed,
}: {
  instrument: string;
  price: number;
  changeAbs: number | null;
  marketClosed: boolean;
}) {
  const t = useTranslations('app');
  const fmt = useReadingFormatters();
  const tone = fmt.changeTone(changeAbs);
  const sign = changeAbs == null || changeAbs === 0 ? '' : changeAbs > 0 ? '+' : '−';
  return (
    <div className="apphead">
      <h1>{fmt.instrument(instrument)}</h1>
      <span className="price mono">{fmt.price(price, instrument)}</span>
      {changeAbs != null && (
        <span
          className={`mono ${tone === 'bull' ? 'up' : tone === 'bear' ? 'dn' : ''}`}
          style={{ fontSize: '12px', fontWeight: 600 }}
        >
          {sign}
          {fmt.price(Math.abs(changeAbs), instrument)}
        </span>
      )}
      <span className="hsp" />
      {marketClosed ? (
        <div className="livebadge" role="status">
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: '50%',
              background: 'var(--faint)',
              flexShrink: 0,
            }}
            aria-hidden
          />
          <span className="mono">{t('chart.marketClosed')}</span>
        </div>
      ) : (
        <div className="livebadge" role="status">
          <span className="dot" aria-hidden />
          <span className="mono">{t('desktop.live')}</span>
        </div>
      )}
    </div>
  );
}

function LegalBar() {
  const t = useTranslations('app');
  return (
    <div className="legalbar">
      <span className="ea">
        <span className="d" aria-hidden />
        {t('desktop.earlyAccess')}
      </span>
      <span className="legal-inline">
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M12 3l8 4v5c0 4-3.4 7.4-8 9-4.6-1.6-8-5-8-9V7l8-4z" />
        </svg>
        {t('desktop.legalInline')}
      </span>
    </div>
  );
}

/* ── Layer pills ──────────────────────────────────────────────────────────── */

function LayerPills({
  layers,
  activeOnly,
  onToggleLayer,
  onToggleMitigated,
}: {
  layers: { ob: boolean; fvg: boolean; liquidity: boolean; breaks: boolean };
  activeOnly: boolean;
  onToggleLayer: (layer: 'ob' | 'fvg' | 'liquidity' | 'breaks', on: boolean) => void;
  onToggleMitigated: (shown: boolean) => void;
}) {
  const t = useTranslations('app');
  // "Mitigées" ON = mitigated/tested zones shown = NOT active-only.
  const mitigatedShown = !activeOnly;
  const single: Array<{
    key: 'ob' | 'fvg' | 'liquidity' | 'breaks';
    label: string;
    sw: string;
    on: boolean;
  }> = [
    { key: 'ob', label: t('layers.ob'), sw: 'var(--ob-l)', on: layers.ob },
    { key: 'fvg', label: t('layers.fvg'), sw: 'var(--fvg-l)', on: layers.fvg },
    { key: 'liquidity', label: t('layers.liquidity'), sw: 'var(--liq)', on: layers.liquidity },
    { key: 'breaks', label: t('layers.breaks'), sw: 'var(--dim)', on: layers.breaks },
  ];
  return (
    <div className="layerrow">
      {single.map((l) => (
        <button
          key={l.key}
          type="button"
          className={`layer${l.on ? '' : ' off'}`}
          aria-pressed={l.on}
          onClick={() => onToggleLayer(l.key, l.on)}
        >
          <span className="sw" style={{ background: l.sw }} aria-hidden />
          {l.label}
        </button>
      ))}
      <button
        type="button"
        className={`layer${mitigatedShown ? '' : ' off'}`}
        aria-pressed={mitigatedShown}
        onClick={() => onToggleMitigated(mitigatedShown)}
      >
        <span className="sw" style={{ background: 'var(--faint)' }} aria-hidden />
        {t('layers.mitigated')}
      </button>
    </div>
  );
}

/* ── Panels ───────────────────────────────────────────────────────────────── */

function NarratedPanel({ conditions }: { conditions: MarketReadingConditions }) {
  const t = useTranslations('app');
  const paragraphs = conditions.description
    .split(/\n+/)
    .map((p) => p.trim())
    .filter(Boolean);
  return (
    <div className="card wide">
      <div className="card-h">
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M4 6h16M4 12h16M4 18h10" />
        </svg>
        <h3>{t('desktop.narratedTitle')}</h3>
        <span className="badge2">{t('desktop.narratedBadge')}</span>
      </div>
      <div className="narr">
        {paragraphs.length > 0 ? (
          paragraphs.map((p, i) => <p key={i}>{p}</p>)
        ) : (
          <p>{conditions.description}</p>
        )}
      </div>
      <div className="narrfoot">
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M9 12l2 2 4-4" />
          <circle cx="12" cy="12" r="9" />
        </svg>
        {t('desktop.narratedFooter')}
      </div>
    </div>
  );
}

function RegimePanel({
  regime,
  structure,
  header,
}: {
  regime: MarketReadingRegime;
  structure: MarketReadingStructure;
  header: MarketReadingHeader;
}) {
  const t = useTranslations('app');
  const tr = useTranslations('reading');
  const fmt = useReadingFormatters();
  const { trends } = useMtfTrends(header.instrument);

  const maturity = deriveTrendMaturity(structure, header);
  const maturityVal =
    maturity?.bars != null ? `≈ ${maturity.bars} ${header.timeframe}` : null;

  const relation = classifyMtfAlignment(trends);
  const hasTrend = MTF_TREND_ORDER.some(({ key }) => trends[key] !== null);
  const alignmentVal = hasTrend
    ? MTF_TREND_ORDER.filter(({ key }) => trends[key] !== null)
        .map(({ key, label }) => `${label} ${mtfTrendGlyph(trends[key]).arrow}`)
        .join(' · ')
    : null;

  const cells: Array<{ k: string; v: string | null }> = [
    { k: t('desktop.reg.trend'), v: fmt.trend(regime.trend).label },
    { k: t('desktop.reg.volatility'), v: fmt.volatility(regime.volatility_observed).label },
    { k: t('desktop.reg.maturity'), v: maturityVal },
    { k: t('desktop.reg.alignment'), v: alignmentVal },
    { k: t('desktop.reg.lastEvent'), v: fmt.regimeLastEvent(structure, header) },
    { k: t('desktop.reg.density'), v: fmt.regimeZoneDensity(structure) },
  ];

  return (
    <div className="card">
      <div className="card-h">
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M3 17l6-6 4 4 8-8" />
          <path d="M21 3v6h-6" />
        </svg>
        <h3>{tr('regime.title')}</h3>
        {relation.disagreement && (
          <span className="badge2" style={{ color: 'var(--liq)' }}>
            {t('desktop.reg.disagreement')}
          </span>
        )}
      </div>
      <div className="reggrid">
        {cells.map((c) => (
          <div className="reg" key={c.k}>
            <div className="k">{c.k}</div>
            <div className="v" style={c.v == null ? { color: 'var(--faint)', fontStyle: 'italic' } : undefined}>
              {c.v ?? tr('regime.unavailable')}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StructurePanel({
  structure,
  instrument,
}: {
  structure: MarketReadingStructure;
  instrument: string;
}) {
  const t = useTranslations('app');
  const tr = useTranslations('reading');
  const fmt = useReadingFormatters();
  const { bos, choch, order_blocks, fair_value_gaps } = structure;

  type Row = { tag: string; tone: 'bull' | 'bear' | 'neu'; desc: string; right: string };
  const rows: Row[] = [];

  if (choch) {
    rows.push({
      tag: `CHOCH ${choch.direction === 'bullish' ? '↑' : '↓'}`,
      tone: choch.direction === 'bullish' ? 'bull' : 'bear',
      desc: `${fmt.direction(choch.direction)} · ${fmt.validationFem(choch.validation_status)}`,
      right: fmt.price(choch.level, instrument),
    });
  }
  if (bos) {
    rows.push({
      tag: `BOS ${bos.direction === 'bullish' ? '↑' : '↓'}`,
      tone: bos.direction === 'bullish' ? 'bull' : 'bear',
      desc: `${fmt.direction(bos.direction)} · ${fmt.validationFem(bos.validation_status)}`,
      right: fmt.price(bos.level, instrument),
    });
  }
  for (const ob of order_blocks.slice(0, 2)) {
    rows.push({
      tag: 'OB',
      tone: 'neu',
      desc: fmt.band(ob.level_low, ob.level_high, instrument),
      right: fmt.obStatus(ob.status),
    });
  }
  for (const fvg of fair_value_gaps.slice(0, 2)) {
    rows.push({
      tag: 'FVG',
      tone: 'neu',
      desc: fmt.band(fvg.level_low, fvg.level_high, instrument),
      right: fmt.fvgStatus(fvg.status),
    });
  }

  return (
    <div className="card">
      <div className="card-h">
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M3 12h4l3-7 4 14 3-7h4" />
        </svg>
        <h3>{tr('structure.title')}</h3>
      </div>
      {rows.length === 0 ? (
        <p className="emptyrow">{tr('structure.empty')}</p>
      ) : (
        rows.map((r, i) => (
          <div className="strow" key={i}>
            <span className={`tagx ${r.tone}`}>{r.tag}</span>
            <span className="d">{r.desc}</span>
            <span className="t">{r.right}</span>
          </div>
        ))
      )}
      <p className="narrfoot" style={{ borderTop: 'none', marginTop: 6, paddingTop: 0 }}>
        {t('desktop.structureFooter')}
      </p>
    </div>
  );
}

function LiquidityPanel({
  structure,
  instrument,
}: {
  structure: MarketReadingStructure;
  instrument: string;
}) {
  const t = useTranslations('app');
  const tr = useTranslations('reading');
  const fmt = useReadingFormatters();
  const pools = structure.liquidity_pools ?? [];

  return (
    <div className="card">
      <div className="card-h">
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M4 8h16M4 12h16M4 16h16" />
        </svg>
        <h3>{tr('structure.liquidityLabel')}</h3>
        <span className="badge2">{t('desktop.liq.badge')}</span>
      </div>
      {pools.length === 0 ? (
        <p className="emptyrow">{t('desktop.liq.empty')}</p>
      ) : (
        pools.map((p) => {
          const lineClass =
            p.status === 'swept' ? 'lline swept' : p.status === 'broken' ? 'lline broken' : 'lline';
          const lstClass = p.status === 'broken' ? 'lst broken' : 'lst';
          const desc =
            p.status === 'swept'
              ? t('desktop.liq.sweptDesc')
              : p.status === 'broken'
                ? t('desktop.liq.brokenDesc')
                : t('desktop.liq.intactDesc');
          return (
            <div className="liqrow" key={p.id}>
              <div className={lstClass}>
                <div className={lineClass} />
                <div className="s">{fmt.liquidityStatus(p.status).label}</div>
              </div>
              <span className="lv">{fmt.price(p.level, instrument)}</span>
              <span className="dd">{desc}</span>
            </div>
          );
        })
      )}
    </div>
  );
}

function NewsPanel({ events }: { events: MarketReadingEvents }) {
  const t = useTranslations('app');
  const fmt = useReadingFormatters();
  const upcoming = events.news_upcoming;

  return (
    <div className="card">
      <div className="card-h">
        <svg viewBox="0 0 24 24" aria-hidden>
          <rect x="3" y="4" width="18" height="17" rx="2" />
          <path d="M3 9h18M8 2v4M16 2v4" />
        </svg>
        <h3>{t('desktop.news.title')}</h3>
        <span className="badge2">{t('desktop.news.badge')}</span>
      </div>
      {upcoming.length === 0 ? (
        <p className="emptyrow">{t('desktop.news.empty')}</p>
      ) : (
        upcoming.map((n, i) => {
          const impact = fmt.impact(n.impact);
          return (
            <div className="newsrow" key={i}>
              <div className="nic">
                <svg viewBox="0 0 24 24" aria-hidden>
                  <circle cx="12" cy="12" r="9" />
                  <path d="M12 7v5l3 2" />
                </svg>
              </div>
              <div className="main3">
                <div className="ev">
                  {n.event} · {fmt.currency(n.currency)}
                </div>
                <div className="mk3">
                  {t('desktop.news.scheduledOn', { currency: fmt.currency(n.currency) })}
                </div>
              </div>
              <span className={`impact ${n.impact === 'high' ? 'high' : 'mid'}`}>{impact.label}</span>
              <span className="when">{fmt.timeToEvent(n.time_to_event_min)}</span>
            </div>
          );
        })
      )}
    </div>
  );
}
