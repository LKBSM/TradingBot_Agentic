'use client';

import * as React from 'react';
import {
  CandlestickSeries,
  ColorType,
  createChart,
  createSeriesMarkers,
  CrosshairMode,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';
import { Maximize2, Minus, Plus } from 'lucide-react';
import { useTheme } from 'next-themes';
import { cn } from '@/lib/utils';
import {
  buildLiveOverlay,
  buildZoneModels,
  curateZones,
  type ZoneModel,
} from '@/lib/chart/zoneLayout';
import { buildStructureMarkers } from '@/lib/chart/structureMarkers';
import type { Candle, MarketReadingStructure } from '@/types/market-reading';

/**
 * Candlestick chart for the reading panel, built on TradingView's Lightweight
 * Charts (Apache-2.0). It draws candles plus SMC overlays read straight from a
 * MarketReadingStructure:
 *
 *   · BOS / CHOCH / retest break levels → horizontal price lines.
 *   · Order Blocks / Fair Value Gaps → LOCALIZED boxes, anchored to the
 *     formation candle (x-start = created_at) and bounded in time: an active
 *     zone runs to the current bar, a tested/mitigated zone stops at its
 *     mitigation point (mitigated_at, read-only). No full-width bands, no
 *     projection into the future.
 *
 * Visual language = "Direction 1" (sober / institutional): muted body colours,
 * hairline strokes, horizontal-only grid, monospace tabular axis numbers, no
 * neon / glow / gradient. Active zones read crisp (visible fill, marked dashed
 * border, label); tested zones recede (near-transparent fill, ghost dashed
 * border, no label) so the two never compete for attention. The chart only
 * restyles + curates what the engine emits — it never adds, recomputes, or
 * reinterprets a structure.
 *
 * Props are typed against the production contract (Candle[] + structure) so the
 * SAME component renders the real engine output once the backend feeds it — only
 * the data source changes, not this component.
 */

export interface ReadingChartProps {
  candles: Candle[];
  structure: MarketReadingStructure;
  /** Instrument code — drives price precision in the overlay labels. */
  instrument: string;
  /**
   * PROTOTYPE — opt-in live tick price (null when the live overlay is off). When
   * set, the chart layers a PROVISIONAL, intra-candle interaction view on top of
   * the candle-CONFIRMED zones: a FVG shrinks toward the still-open side as the
   * price eats into it, and an active OB is flagged "en test". This NEVER mutates
   * the confirmed structure and NEVER touches BOS/CHOCH; a price retreat reverts
   * the provisional view cleanly. Confirmation still happens only at candle close.
   */
  livePrice?: number | null;
  className?: string;
}

/** Pixel rect for a rendered, localized zone box (recomputed on scale / resize). */
interface ZoneRect {
  id: string;
  kind: 'ob' | 'fvg';
  left: number;
  width: number;
  top: number;
  height: number;
  label: string;
  tested: boolean;
  /** Provisional: an ACTIVE order block the live price is currently inside. */
  inTestLive: boolean;
}

/** Pixel rect for a PROVISIONAL live FVG-fill front (drawn inside its box). */
interface LiveFvgRect {
  id: string;
  left: number;
  width: number;
  top: number;
  height: number;
}

/** Quantise to ½px so sub-pixel float noise never triggers a re-render. */
const qpx = (n: number) => Math.round(n * 2) / 2;

/**
 * Geometry-equal guard for the rAF redraw loop: true when both lists hold the
 * same boxes at the same (½px-rounded) pixel rects. Lets the loop run every
 * frame (so zones stay glued to the price axis) while keeping React idle until
 * a box actually moves.
 */
function rectsEqual(a: ZoneRect[], b: ZoneRect[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i]!;
    const y = b[i]!;
    if (
      x.id !== y.id ||
      x.kind !== y.kind ||
      x.tested !== y.tested ||
      x.inTestLive !== y.inTestLive ||
      qpx(x.left) !== qpx(y.left) ||
      qpx(x.width) !== qpx(y.width) ||
      qpx(x.top) !== qpx(y.top) ||
      qpx(x.height) !== qpx(y.height)
    ) {
      return false;
    }
  }
  return true;
}

/** Geometry-equal guard for the provisional live FVG-front rects. */
function liveRectsEqual(a: LiveFvgRect[], b: LiveFvgRect[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i]!;
    const y = b[i]!;
    if (
      x.id !== y.id ||
      qpx(x.left) !== qpx(y.left) ||
      qpx(x.width) !== qpx(y.width) ||
      qpx(x.top) !== qpx(y.top) ||
      qpx(x.height) !== qpx(y.height)
    ) {
      return false;
    }
  }
  return true;
}

/** Candle bodies — muted bull / bear; wick + border share the body colour. */
const CANDLE = { bull: '#2F9E78', bear: '#C2693E' };

/** Break-level line colours — sober, distinguishable, hairline. */
const LEVEL = {
  bos: '#8B95A7',
  choch: '#8E84B0',
  retest: '#6E84B0',
};

/**
 * Sober zone palette, per Direction 1. One base RGB per kind; the alpha encodes
 * the active/tested hierarchy (crisp vs ghost) so a tested box never competes
 * with an active one. No neon / glow / gradient.
 */
const ZONE_RGB = {
  ob: '139, 149, 167', // #8B95A7
  fvg: '110, 132, 176', // #6E84B0
} as const;

const ZONE_LABEL = {
  ob: '#9AA4B8',
  fvg: '#6E84B0',
} as const;

/** Short type code shown INSIDE every box so OB vs FVG is always identifiable. */
const ZONE_CODE = {
  ob: 'OB',
  fvg: 'FVG',
} as const;

/** Fill / border alpha by state — active reads, tested recedes (~0.05 fill). */
const ZONE_ALPHA = {
  active: { fill: 0.12, border: 0.45 },
  tested: { fill: 0.05, border: 0.18 },
} as const;

/**
 * PROVISIONAL / live accent — deliberately a DIFFERENT hue (warm amber) from the
 * cool slate/blue confirmed palette, so an intra-candle "in progress" state can
 * never be mistaken for a candle-confirmed one. Used for the live FVG-fill front
 * and the OB "en test" flag.
 */
const LIVE_RGB = '201, 162, 39'; // #C9A227 — warm amber
const LIVE_COLOR = '#C9A227';

/**
 * Theme-resolved palette. Lightweight-charts paints onto a canvas, so it needs
 * concrete colour strings (CSS `var(--token)` does not resolve there). These
 * values mirror the app tokens — `--border` for the grid, `--muted-foreground`
 * for axis text — at both light and dark, so the chart tracks the app theme.
 */
function palette(isDark: boolean) {
  return isDark
    ? {
        axisText: 'hsl(215, 20%, 65%)', // --muted-foreground (dark) → tertiary axis
        grid: 'hsla(217, 33%, 17%, 0.4)', // --border (dark) @ 0.4
        scaleBorder: 'hsla(217, 33%, 17%, 0.6)',
        crosshair: 'hsla(215, 20%, 65%, 0.45)',
        crosshairLabel: '#3f3f46',
      }
    : {
        axisText: 'hsl(215, 16%, 47%)', // --muted-foreground (light) → tertiary axis
        grid: 'hsla(214, 32%, 91%, 0.4)', // --border (light) @ 0.4
        scaleBorder: 'hsla(214, 32%, 91%, 0.8)',
        crosshair: 'hsla(215, 16%, 47%, 0.40)',
        crosshairLabel: '#52525b',
      };
}

const MONO_FONT = "'ui-monospace', 'SFMono-Regular', 'Menlo', monospace";

export function ReadingChart({
  candles,
  structure,
  instrument,
  livePrice = null,
  className,
}: ReadingChartProps) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';

  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const chartRef = React.useRef<IChartApi | null>(null);
  const seriesRef = React.useRef<ISeriesApi<'Candlestick'> | null>(null);
  const markersRef = React.useRef<ISeriesMarkersPluginApi<Time> | null>(null);

  const [zoneRects, setZoneRects] = React.useState<ZoneRect[]>([]);
  const [liveFvgRects, setLiveFvgRects] = React.useState<LiveFvgRect[]>([]);

  // PROTOTYPE — provisional intra-candle interaction overlay derived from the
  // latest tick. Recomputed from the CURRENT price only (never persisted), so a
  // price retreat reverts it cleanly. Only describes FVG fill / OB touch — never
  // detection, never BOS/CHOCH.
  const liveOverlay = React.useMemo(
    () => buildLiveOverlay(structure, livePrice),
    [structure, livePrice],
  );
  const liveActive =
    typeof livePrice === 'number' &&
    Number.isFinite(livePrice) &&
    (liveOverlay.fvgFronts.length > 0 || liveOverlay.obInTest.size > 0);

  // Current bar = last candle: its close anchors the proximity curation, its
  // time bounds the right edge of an active (unmitigated) box.
  const lastCandle = candles.length > 0 ? candles[candles.length - 1] : null;

  // Curated zone models (localized + bounded). Tested first so active boxes
  // layer on top when rendered; consumed zones are already excluded upstream.
  const zones = React.useMemo<(ZoneModel & { tested: boolean })[]>(() => {
    const models = buildZoneModels(structure);
    const price = lastCandle?.close ?? 0;
    const { active, tested } = curateZones(models, price);
    return [...tested, ...active];
  }, [structure, lastCandle?.close]);

  // ── Create the chart once; recreate only if the theme changes. ──────────────
  React.useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const p = palette(isDark);

    const chart = createChart(container, {
      // Responsive: tracks the container box (paired with the ResizeObserver
      // below so the chart never feels cramped on small / rotated screens).
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: p.axisText,
        fontFamily: MONO_FONT, // monospace → tabular axis numbers.
        fontSize: 11,
        attributionLogo: true, // TradingView attribution (Apache-2.0 licence).
      },
      grid: {
        // Horizontal hairlines only — vertical lines off for a calmer canvas.
        vertLines: { visible: false },
        horzLines: { color: p.grid, style: LineStyle.Solid },
      },
      crosshair: {
        mode: CrosshairMode.Magnet,
        vertLine: {
          color: p.crosshair,
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: p.crosshairLabel,
        },
        horzLine: {
          color: p.crosshair,
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: p.crosshairLabel,
        },
      },
      rightPriceScale: { borderColor: p.scaleBorder },
      timeScale: {
        borderColor: p.scaleBorder,
        timeVisible: true,
        secondsVisible: false,
      },
      // ── Interaction: fluid pan / zoom on mouse AND touch. ──
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        mouseWheel: true,
        pinch: true, // pinch-zoom on touch.
        axisPressedMouseMove: true,
        axisDoubleClickReset: true,
      },
      kineticScroll: { touch: true, mouse: false },
      // trackingMode defaults to OnNextTap → tap-to-read crosshair on mobile.
    });

    const series = chart.addSeries(CandlestickSeries, {
      upColor: CANDLE.bull,
      downColor: CANDLE.bear,
      // Wick + border share the body colour (no contrasting outline).
      borderUpColor: CANDLE.bull,
      borderDownColor: CANDLE.bear,
      wickUpColor: CANDLE.bull,
      wickDownColor: CANDLE.bear,
      // Current-price line: hairline, dashed, colour follows the last move.
      priceLineVisible: true,
      priceLineWidth: 1,
      priceLineStyle: LineStyle.Dashed,
      // Empty colour → inherits the up/down colour of the last candle, so the
      // axis badge reads green when up, terracotta when down (white label text).
      priceLineColor: '',
      lastValueVisible: true,
    });

    chartRef.current = chart;
    seriesRef.current = series;
    // Markers plugin for the BOS/CHOCH break history (set in the data effect).
    markersRef.current = createSeriesMarkers(series, []);

    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      markersRef.current = null;
    };
  }, [isDark]);

  // ── Push candle data + structure price lines; fit content. ──────────────────
  React.useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series) return;

    series.setData(
      candles.map((c) => ({
        time: c.time as UTCTimestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      })),
    );

    // BOS / CHOCH break-history markers (read-only, descriptive). One arrow per
    // detected break over the window — fixes the "sous-surfaçage" where only the
    // last bar's break ever showed. Lightweight-charts ignores markers outside
    // the loaded candle range, so older breaks simply don't draw (graceful).
    markersRef.current?.setMarkers(buildStructureMarkers(structure));

    // Horizontal break-level price lines (BOS / CHOCH / retest) — hairline.
    const priceLines = [
      structure.bos && {
        price: structure.bos.level,
        color: LEVEL.bos,
        title: 'BOS',
      },
      structure.choch && {
        price: structure.choch.level,
        color: LEVEL.choch,
        title: 'CHOCH',
      },
      structure.retest_in_progress && {
        price: structure.retest_in_progress.level,
        color: LEVEL.retest,
        title: 'Retest',
      },
    ].filter((l): l is { price: number; color: string; title: string } =>
      Boolean(l),
    );

    const created = priceLines.map((l) =>
      series.createPriceLine({
        price: l.price,
        color: l.color,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: l.title,
      }),
    );

    chart.timeScale().fitContent();

    return () => {
      for (const line of created) series.removePriceLine(line);
    };
  }, [candles, structure]);

  // ── Keep zone rectangles IN PHASE with the chart canvas. ────────────────────
  // The boxes are an HTML overlay positioned from priceToCoordinate /
  // timeToCoordinate. The time axis emits a range-change event, but a VERTICAL
  // PRICE-AXIS drag (handleScale.axisPressedMouseMove) emits nothing — so the
  // old "subscribe to the time range + ResizeObserver" wiring never recomputed
  // on price scaling: boxes stayed frozen at stale Y, then snapped on the next
  // time event = the jitter. Fix: recompute every animation frame so the
  // overlay tracks the canvas for ALL scale changes (price drag, time pan/zoom,
  // autoscale, resize), gated by a geometry guard so React only re-renders when
  // a box truly moves.
  React.useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    const container = containerRef.current;
    if (!chart || !series || !container) return;

    const timeScale = chart.timeScale();

    // Snap a target time to the nearest candle so timeToCoordinate (which maps
    // data points) always resolves, then clamp into the plot area. Boxes never
    // overrun the price-scale gutter and never project past the current bar.
    const candleTimes = candles.map((c) => c.time as number);
    const lastTime = candleTimes.length ? candleTimes[candleTimes.length - 1]! : null;
    const snapToCandle = (sec: number): number | null => {
      if (!candleTimes.length) return null;
      let best = candleTimes[0]!;
      let bestD = Math.abs(best - sec);
      for (const t of candleTimes) {
        const d = Math.abs(t - sec);
        if (d < bestD) {
          best = t;
          bestD = d;
        }
      }
      return best;
    };

    const computeRects = (): { rects: ZoneRect[]; live: LiveFvgRect[] } => {
      const plotRight = Math.max(
        0,
        container.clientWidth - chart.priceScale('right').width(),
      );
      const clampX = (x: number) => Math.min(Math.max(x, 0), plotRight);
      const xAt = (sec: number): number | null => {
        const snapped = snapToCandle(sec);
        if (snapped === null) return null;
        const c = timeScale.timeToCoordinate(snapped as UTCTimestamp);
        return c === null ? null : clampX(c);
      };

      // Provisional live fronts, keyed by FVG id (recomputed each frame).
      const liveFronts = new Map(liveOverlay.fvgFronts.map((f) => [f.id, f]));

      const rects: ZoneRect[] = [];
      const live: LiveFvgRect[] = [];
      for (const z of zones) {
        const yHigh = series.priceToCoordinate(z.high);
        const yLow = series.priceToCoordinate(z.low);
        if (yHigh === null || yLow === null) continue;
        const top = Math.min(yHigh, yLow);
        const height = Math.max(2, Math.abs(yLow - yHigh));

        // x-start = formation; x-end = mitigation point for a tested zone, else
        // the current bar for an active one (no future projection).
        const xStart = xAt(z.createdSec);
        const endSec = z.tested && z.mitigatedSec !== null ? z.mitigatedSec : lastTime;
        const xEnd = endSec === null ? null : xAt(endSec);
        if (xStart === null || xEnd === null) continue;
        const left = Math.min(xStart, xEnd);
        const width = Math.max(2, Math.abs(xEnd - xStart));

        rects.push({
          id: z.id,
          kind: z.kind,
          left,
          width,
          top,
          height,
          label: z.label,
          tested: z.tested,
          // Provisional: an ACTIVE order block the live price is inside right now.
          inTestLive: z.kind === 'ob' && liveOverlay.obInTest.has(z.id),
        });

        // Provisional live FVG-fill front: the still-open band drawn INSIDE the
        // confirmed FVG box (warm-amber accent), tied to the same x-span so it
        // reads as "the gap being eaten right now". Shrinks as price penetrates;
        // disappears when price retreats (front absent from the overlay).
        const front = z.kind === 'fvg' ? liveFronts.get(z.id) : undefined;
        if (front) {
          const fHigh = series.priceToCoordinate(front.high);
          const fLow = series.priceToCoordinate(front.low);
          if (fHigh !== null && fLow !== null) {
            live.push({
              id: z.id,
              left,
              width,
              top: Math.min(fHigh, fLow),
              height: Math.max(2, Math.abs(fLow - fHigh)),
            });
          }
        }
      }
      return { rects, live };
    };

    // Animation-frame loop: recompute in phase with the canvas, commit to React
    // only when the geometry changes (rectsEqual guard) so an idle chart costs
    // a cheap coordinate read per frame and zero re-renders.
    let prev: ZoneRect[] = [];
    let prevLive: LiveFvgRect[] = [];
    let raf = 0;
    const tick = () => {
      const { rects, live } = computeRects();
      if (!rectsEqual(prev, rects)) {
        prev = rects;
        setZoneRects(rects);
      }
      if (!liveRectsEqual(prevLive, live)) {
        prevLive = live;
        setLiveFvgRects(live);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
    };
  }, [zones, candles, liveOverlay]);

  // ── Discreet zoom / fit controls (mouse + ≥44px touch targets). ─────────────
  const zoom = React.useCallback((factor: number) => {
    const ts = chartRef.current?.timeScale();
    if (!ts) return;
    const range = ts.getVisibleLogicalRange();
    if (!range) return;
    const center = (range.from + range.to) / 2;
    const half = ((range.to - range.from) / 2) * factor;
    ts.setVisibleLogicalRange({ from: center - half, to: center + half });
  }, []);

  const fit = React.useCallback(() => {
    chartRef.current?.timeScale().fitContent();
  }, []);

  return (
    <div className={cn('relative w-full', className)}>
      <div
        ref={containerRef}
        className="h-[280px] w-full sm:h-[340px]"
        role="img"
        aria-label={`Graphique en chandeliers ${instrument} avec zones Order Block, Fair Value Gap et niveaux de cassure`}
      />

      {/* Localized OB / FVG boxes, layered over the chart canvas. Active boxes
          read crisp; tested boxes recede (ghost fill/border). Every box carries
          a short OB / FVG type code at its top-left so the kind is always
          identifiable — crisp on active, dimmer + smaller on tested. An active
          OB the live price is inside gets a warm-amber "en test" accent (a
          PROVISIONAL, intra-candle state — distinct hue from the confirmed
          palette so it's never read as a candle-confirmed outcome). */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        {zoneRects.map((r) => {
          const rgb = ZONE_RGB[r.kind];
          const a = r.tested ? ZONE_ALPHA.tested : ZONE_ALPHA.active;
          return (
            <div
              key={`${r.kind}:${r.id}`}
              className="absolute"
              style={{
                left: r.left,
                width: r.width,
                top: r.top,
                height: r.height,
                backgroundColor: `rgba(${rgb}, ${a.fill})`,
                border: r.inTestLive
                  ? `1px solid rgba(${LIVE_RGB}, 0.85)`
                  : `1px dashed rgba(${rgb}, ${a.border})`,
                borderRadius: 1,
              }}
            >
              <span
                className={cn(
                  'absolute left-1 top-0 whitespace-nowrap font-medium leading-tight tabular-nums',
                  r.tested ? 'text-[9px] opacity-70' : 'text-[10px]',
                )}
                style={{ color: ZONE_LABEL[r.kind] }}
                title={r.label}
              >
                {ZONE_CODE[r.kind]}
              </span>
              {r.inTestLive && (
                <span
                  className="absolute right-1 top-0 whitespace-nowrap text-[9px] font-semibold leading-tight"
                  style={{ color: LIVE_COLOR }}
                  title="Order Block en cours de test — provisoire, intra-bougie (confirmé seulement à la clôture)"
                >
                  • en test
                </span>
              )}
            </div>
          );
        })}

        {/* PROVISIONAL live FVG-fill front — the still-open band right now, drawn
            inside its confirmed FVG box in warm amber. Descriptive: the price is
            literally there. Shrinks live as price fills the gap; vanishes on a
            retreat. Never confirmed — confirmation lands at candle close. */}
        {liveFvgRects.map((r) => (
          <div
            key={`live-fvg:${r.id}`}
            className="absolute"
            style={{
              left: r.left,
              width: r.width,
              top: r.top,
              height: r.height,
              backgroundColor: `rgba(${LIVE_RGB}, 0.16)`,
              border: `1px solid rgba(${LIVE_RGB}, 0.8)`,
              borderRadius: 1,
            }}
            title="Comblement du FVG en cours — provisoire, intra-bougie (confirmé seulement à la clôture)"
          >
            <span
              className="absolute right-1 top-0 whitespace-nowrap text-[9px] font-semibold leading-tight"
              style={{ color: LIVE_COLOR }}
            >
              comblement live
            </span>
          </div>
        ))}
      </div>

      {/* Live-mode badge — makes the PROVISIONAL state explicit and honest. Only
          shown when a tick is actually driving a provisional interaction. */}
      {liveActive && (
        <div
          className="pointer-events-none absolute right-2 top-2 flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold"
          style={{
            color: LIVE_COLOR,
            backgroundColor: `rgba(${LIVE_RGB}, 0.12)`,
            border: `1px solid rgba(${LIVE_RGB}, 0.5)`,
          }}
          title="Interaction de zones EN DIRECT (provisoire, intra-bougie). Les cassures BOS/CHOCH et les invalidations ne sont confirmées qu'à la clôture de bougie."
          role="status"
          aria-live="polite"
        >
          <span
            className="inline-block h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: LIVE_COLOR }}
            aria-hidden
          />
          EN DIRECT · provisoire
        </div>
      )}

      {/* Sober pan/zoom controls — visually light, ≥44px tap zone on mobile. */}
      <div className="absolute bottom-2 left-2 flex gap-1">
        <ChartControl label="Zoom avant" onClick={() => zoom(0.7)}>
          <Plus className="h-4 w-4" aria-hidden />
        </ChartControl>
        <ChartControl label="Zoom arrière" onClick={() => zoom(1.4)}>
          <Minus className="h-4 w-4" aria-hidden />
        </ChartControl>
        <ChartControl label="Ajuster le graphique" onClick={fit}>
          <Maximize2 className="h-4 w-4" aria-hidden />
        </ChartControl>
      </div>
    </div>
  );
}

/** A single sober chart control: hairline border, ≥44px tap target on mobile. */
function ChartControl({
  label,
  onClick,
  children,
}: {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={cn(
        'flex h-11 w-11 items-center justify-center rounded-md border border-border/60',
        'bg-background/70 text-muted-foreground backdrop-blur-sm',
        'transition-colors hover:text-foreground',
        'sm:h-8 sm:w-8',
      )}
    >
      {children}
    </button>
  );
}
