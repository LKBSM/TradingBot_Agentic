'use client';

import * as React from 'react';
import {
  CandlestickSeries,
  ColorType,
  createChart,
  CrosshairMode,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from 'lightweight-charts';
import { Maximize2, Minus, Plus } from 'lucide-react';
import { useTheme } from 'next-themes';
import { cn } from '@/lib/utils';
import type { Candle, MarketReadingStructure } from '@/types/market-reading';

/**
 * Candlestick chart for the reading panel, built on TradingView's Lightweight
 * Charts (Apache-2.0). It draws candles plus SMC overlays read straight from a
 * MarketReadingStructure:
 *
 *   · BOS / CHOCH / retest break levels → horizontal price lines.
 *   · Order Blocks  → slate shaded price bands.
 *   · Fair Value Gaps → blue shaded price bands.
 *
 * Visual language = "Direction 1" (sober / institutional): muted body colours,
 * hairline strokes, horizontal-only grid, monospace tabular axis numbers, no
 * neon / glow / gradient. Only *active* structures are surfaced (faded when the
 * engine marks them inactive) — the styling restyles what the engine emits, it
 * never adds, hides, or reinterprets a structure.
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
  className?: string;
}

/** A shaded price band overlay (Order Block / Fair Value Gap). */
interface ZoneOverlay {
  id: string;
  kind: 'ob' | 'fvg';
  high: number;
  low: number;
  label: string;
  faded: boolean;
}

/** Pixel rect for a rendered zone (recomputed on scale / resize changes). */
interface ZoneRect {
  id: string;
  kind: 'ob' | 'fvg';
  top: number;
  height: number;
  label: string;
  faded: boolean;
}

/** Candle bodies — muted bull / bear; wick + border share the body colour. */
const CANDLE = { bull: '#2F9E78', bear: '#C2693E' };

/** Break-level line colours — sober, distinguishable, hairline. */
const LEVEL = {
  bos: '#8B95A7',
  choch: '#8E84B0',
  retest: '#6E84B0',
};

/** Sober zone palette (fill / dashed border / label), per Direction 1. */
const ZONE_STYLE = {
  ob: {
    fill: 'rgba(139, 149, 167, 0.10)', // #8B95A7
    border: 'rgba(139, 149, 167, 0.40)',
    label: '#9AA4B8',
  },
  fvg: {
    fill: 'rgba(110, 132, 176, 0.10)', // #6E84B0
    border: 'rgba(110, 132, 176, 0.40)',
    label: '#6E84B0',
  },
} as const;

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
  className,
}: ReadingChartProps) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';

  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const chartRef = React.useRef<IChartApi | null>(null);
  const seriesRef = React.useRef<ISeriesApi<'Candlestick'> | null>(null);

  const [zoneRects, setZoneRects] = React.useState<ZoneRect[]>([]);

  // Zone overlays (OB + FVG) derived from the structure — stable per structure.
  const zones = React.useMemo<ZoneOverlay[]>(() => {
    const out: ZoneOverlay[] = [];
    for (const ob of structure.order_blocks) {
      out.push({
        id: ob.id,
        kind: 'ob',
        high: ob.level_high,
        low: ob.level_low,
        label: 'Order Block',
        faded: ob.status !== 'active',
      });
    }
    for (const fvg of structure.fair_value_gaps) {
      out.push({
        id: fvg.id,
        kind: 'fvg',
        high: fvg.level_high,
        low: fvg.level_low,
        label: 'Fair Value Gap',
        faded: fvg.status === 'filled',
      });
    }
    return out;
  }, [structure]);

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

    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
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

  // ── Recompute zone rectangles on scale / resize changes. ────────────────────
  React.useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    const container = containerRef.current;
    if (!chart || !series || !container) return;

    const recompute = () => {
      const rects: ZoneRect[] = [];
      for (const z of zones) {
        const yHigh = series.priceToCoordinate(z.high);
        const yLow = series.priceToCoordinate(z.low);
        if (yHigh === null || yLow === null) continue;
        const top = Math.min(yHigh, yLow);
        const height = Math.max(2, Math.abs(yLow - yHigh));
        rects.push({
          id: z.id,
          kind: z.kind,
          top,
          height,
          label: z.label,
          faded: z.faded,
        });
      }
      setZoneRects(rects);
    };

    recompute();
    const timeScale = chart.timeScale();
    timeScale.subscribeVisibleLogicalRangeChange(recompute);
    const ro = new ResizeObserver(recompute);
    ro.observe(container);

    return () => {
      timeScale.unsubscribeVisibleLogicalRangeChange(recompute);
      ro.disconnect();
    };
  }, [zones, candles]);

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

      {/* Shaded OB / FVG bands, layered over the chart canvas. */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        {zoneRects.map((r) => {
          const style = ZONE_STYLE[r.kind];
          return (
            <div
              key={`${r.kind}:${r.id}`}
              className="absolute left-0 right-[64px]"
              style={{
                top: r.top,
                height: r.height,
                backgroundColor: style.fill,
                borderTop: `1px dashed ${style.border}`,
                borderBottom: `1px dashed ${style.border}`,
                opacity: r.faded ? 0.4 : 1,
              }}
            >
              <span
                className="absolute left-1 top-0 text-[10px] font-normal leading-tight tabular-nums"
                style={{ color: style.label }}
              >
                {r.label}
              </span>
            </div>
          );
        })}
      </div>

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
