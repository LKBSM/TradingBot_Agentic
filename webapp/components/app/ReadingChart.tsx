'use client';

import * as React from 'react';
import {
  CandlestickSeries,
  ColorType,
  createChart,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from 'lightweight-charts';
import { useTheme } from 'next-themes';
import { cn } from '@/lib/utils';
import type { Candle } from '@/lib/mockReadings';
import type { MarketReadingStructure } from '@/types/market-reading';

/**
 * Candlestick chart for the reading panel, built on TradingView's Lightweight
 * Charts (Apache-2.0). It draws candles plus SMC overlays read straight from a
 * MarketReadingStructure:
 *
 *   · BOS / CHOCH / retest break levels → horizontal price lines.
 *   · Order Blocks  → amber shaded price bands.
 *   · Fair Value Gaps → blue shaded price bands.
 *
 * Props are typed against the production contract (Candle[] + structure) so the
 * SAME component renders the real engine output once the backend feeds it — only
 * the data source changes, not this component. (Candle data is mock today; see
 * lib/mockReadings.ts.)
 *
 * Beta scope (per brief): candles + OB + FVG + break levels. Finer overlays
 * (uncertainty band, bias marker polish) are V1.1.
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

const COLORS = {
  bull: '#16a34a',
  bear: '#dc2626',
  bos: '#c9a961',
  choch: '#8b5cf6',
  retest: '#0ea5e9',
};

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

    const chart = createChart(container, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: isDark ? '#a1a1aa' : '#52525b',
        fontFamily: 'inherit',
        attributionLogo: true, // TradingView attribution (Apache-2.0 licence).
      },
      grid: {
        vertLines: { color: isDark ? '#27272a' : '#f4f4f5' },
        horzLines: { color: isDark ? '#27272a' : '#f4f4f5' },
      },
      rightPriceScale: { borderColor: isDark ? '#3f3f46' : '#e4e4e7' },
      timeScale: {
        borderColor: isDark ? '#3f3f46' : '#e4e4e7',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const series = chart.addSeries(CandlestickSeries, {
      upColor: COLORS.bull,
      downColor: COLORS.bear,
      borderUpColor: COLORS.bull,
      borderDownColor: COLORS.bear,
      wickUpColor: COLORS.bull,
      wickDownColor: COLORS.bear,
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

    // Horizontal break-level price lines (BOS / CHOCH / retest).
    const priceLines = [
      structure.bos && {
        price: structure.bos.level,
        color: COLORS.bos,
        title: 'BOS',
      },
      structure.choch && {
        price: structure.choch.level,
        color: COLORS.choch,
        title: 'CHOCH',
      },
      structure.retest_in_progress && {
        price: structure.retest_in_progress.level,
        color: COLORS.retest,
        title: 'Retest',
      },
    ].filter((l): l is { price: number; color: string; title: string } =>
      Boolean(l),
    );

    const created = priceLines.map((l) =>
      series.createPriceLine({
        price: l.price,
        color: l.color,
        lineWidth: 2,
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
        {zoneRects.map((r) => (
          <div
            key={`${r.kind}:${r.id}`}
            className={cn(
              'absolute left-0 right-[64px] border-y',
              r.kind === 'ob'
                ? 'border-[#c9a961]/50 bg-[#c9a961]/10'
                : 'border-[#0ea5e9]/50 bg-[#0ea5e9]/10',
              r.faded && 'opacity-40',
            )}
            style={{ top: r.top, height: r.height }}
          >
            <span
              className={cn(
                'absolute left-1 top-0 text-[10px] font-medium leading-tight',
                r.kind === 'ob' ? 'text-[#c9a961]' : 'text-[#0ea5e9]',
              )}
            >
              {r.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
