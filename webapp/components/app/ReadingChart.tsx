'use client';

import * as React from 'react';
import {
  CandlestickSeries,
  ColorType,
  createChart,
  createSeriesMarkers,
  CrosshairMode,
  LineStyle,
  TickMarkType,
  type AutoscaleInfo,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type MouseEventParams,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';
import { Droplets, Maximize2, Minus, Plus } from 'lucide-react';
import { useTheme } from 'next-themes';
import { useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import { formatLocalDayHm, formatLocalHm, localTimeLabel } from '@/lib/time/localTime';
import {
  applyZoneVisibility,
  buildLiveOverlay,
  buildZoneModels,
  curateZones,
  filterZoneModels,
  zoneRightAnchor,
  type ZoneModel,
} from '@/lib/chart/zoneLayout';
import type {
  ChartFilter,
  ChartLayers,
  FocusCommand,
  ReferenceLevel,
} from '@/lib/chart/viewActions';
import { DEFAULT_CHART_VIEW } from '@/lib/chart/viewActions';
import { buildStructureMarkers } from '@/lib/chart/structureMarkers';
import { buildLiquidityLines } from '@/lib/chart/liquidityLines';
import { computeReferenceViewRange } from '@/lib/chart/referenceView';
import {
  ZoneOverlayPrimitive,
  liquidityColor,
  LIVE_RGB,
  LIVE_COLOR,
  type ZoneOverlayData,
} from '@/lib/chart/zoneOverlayPrimitive';
import { isPlausibleTick, isValidBar } from '@/lib/chart/sanitize';
import { badgeLabelKey } from '@/lib/market-reading/status';
import type { Candle, MarketReadingStructure, MarketState } from '@/types/market-reading';

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
  /** Displayed timeframe (M15/H1/H4) — sizes the live forming candle's bucket. */
  timeframe?: string | null;
  /**
   * PROTOTYPE — opt-in live tick price (null when the live overlay is off). When
   * set, the chart shows a TradingView-style live view ON TOP of the candle-
   * CONFIRMED data: the forming (current) candle grows with each tick, and the
   * zone-interaction overlay updates (a FVG shrinks toward the still-open side as
   * the price eats into it; an active OB is flagged "en test"). This NEVER mutates
   * the confirmed/closed candles or structure and NEVER touches BOS/CHOCH; a price
   * retreat reverts the provisional view cleanly. Confirmation is candle close only.
   */
  livePrice?: number | null;
  /** Feed epoch (seconds) of the live tick — buckets the forming candle. */
  liveTs?: number | null;
  /**
   * Descriptive session state. When true (spot FX / gold outside trading hours),
   * the chart shows a neutral "Marché fermé" badge and NEVER the "EN DIRECT"
   * live badge — the app must not claim to be live when it isn't. Display-only;
   * detection is untouched.
   */
  marketClosed?: boolean;
  /**
   * MC-1 server market status. When present it names the badge (closed / daily
   * pause / data delayed) instead of the generic "Marché fermé".
   */
  marketStatusState?: MarketState | null;
  /**
   * DISPLAY-ONLY view state, driven by the M.I.A Agent chat (or left at the
   * defaults). These change ONLY what the chart renders / how it frames — never
   * the detection data or a zone's geometry:
   *   · layers          — which overlay families are drawn (FVG / OB / breaks).
   *   · filter          — which DETECTED zones are shown (active / size / proximity).
   *   · focus           — a one-shot framing command (zone / price / fit).
   *   · highlightZoneId  — a detected zone to emphasise visually.
   *   · hiddenZoneIds    — detected zones removed from the display by id (reversible).
   *   · isolatedZoneIds  — when set, show ONLY these detected zones (null = all).
   * All optional; omitting them yields the exact pre-existing behaviour.
   */
  layers?: ChartLayers;
  filter?: ChartFilter;
  focus?: FocusCommand | null;
  highlightZoneId?: string | null;
  /**
   * A calendar-derived reference marker to trace (day/week open, prev extreme),
   * or null. Rendered as a distinct SOLID accent price line with an axis label —
   * visually separate from detected zones, liquidity pills and dashed break
   * lines. It is NOT a detected zone and travels on its own channel, so it never
   * touches the zone id-lock. See `ReferenceLevel`.
   */
  referenceLevel?: ReferenceLevel | null;
  /** Called when the user clicks the highlighted (blue) zone to deselect it. */
  onClearHighlight?: () => void;
  hiddenZoneIds?: readonly string[];
  isolatedZoneIds?: readonly string[] | null;
  className?: string;
  /**
   * Height utility for the plot container. Defaults to a fluid, viewport-scaled
   * height for the /app workspace so the plot isn't crushed on small screens;
   * the landing hero passes a fixed height to keep the marketing layout stable.
   */
  heightClassName?: string;
}

/** Timeframe → bar length in seconds (for the live forming candle bucket). */
const TF_SECONDS: Record<string, number> = {
  M15: 900,
  H1: 3600,
  H4: 14400,
};

/** Candle bodies — muted bull / bear fallbacks (live values come from the
 *  `--bull` / `--bear` design tokens read in palette()). */
const CANDLE = { bull: '#2F9E78', bear: '#C2693E' };

/** Break-level line colours — sober, distinguishable, hairline. */
const LEVEL = {
  bos: '#8B95A7',
  choch: '#8E84B0',
  retest: '#6E84B0',
};

/** localStorage key for the "intact pockets only" display toggle. */
const LIQUIDITY_INTACT_ONLY_KEY = 'mia.chart.liquidityIntactOnly';

/** Number of recent bars framed when centring on the current price. */
const FOCUS_PRICE_BARS = 40;

/**
 * Default number of most-recent bars shown when the chart first opens. We
 * right-anchor to these (with a little breathing room on the right) instead of
 * fitContent()-ing the whole 200-bar history — so the chart opens on current
 * price action at a readable zoom, not squeezed onto the oldest candle. The
 * user can still scroll left for history or hit "Ajuster" to fit everything.
 */
const DEFAULT_VISIBLE_BARS = 90;
const INITIAL_RIGHT_PAD_BARS = 3;

/**
 * Minimum vertical (price) span the auto-scale is allowed to collapse to,
 * expressed as a fraction of the visible mid price (~0.3%). Lightweight-charts
 * fits the price axis to the visible candles' min/max; when the market is closed
 * (week-end / holiday) the last bars barely move, so that range is a handful of
 * ticks and the axis blows those micro-candles up to fill the whole height — the
 * "zoom extrême" symptom. This floor pads the auto-range to a sensible minimum
 * so a flat window reads as small candles, not magnified ones. It ONLY kicks in
 * when the real range is smaller than the floor; a normal week already exceeds
 * it, so weekday framing is untouched. Display-only — the data is never altered.
 */
const MIN_VISIBLE_RANGE_FRAC = 0.003;

// RG-1d: when a calendar reference level is traced, the vertical viewport is
// widened to hold BOTH the level and the current price, plus this margin on each
// side, so neither sits on the very edge and the structure between them stays
// readable (that context is what makes the trace useful — never a tight zoom).

/**
 * Theme-resolved palette. Lightweight-charts paints onto a canvas, so it needs
 * concrete colour strings (CSS `var(--token)` does not resolve there). We read
 * the LIVE token values off <html> via getComputedStyle so the chart tracks
 * whichever of the four themes is active — structural chrome from `--border` /
 * `--muted-foreground`, and the candle bull/bear from the reserved
 * `--sentinel-*` state tokens (colour = meaning). Called inside a client effect,
 * re-keyed on the resolved theme, so switching between two dark themes repaints.
 */
// The reference literal tokens (--bull #37b98c, --line rgba(…), --acc #4d9de0)
// store ready-to-use CSS colour strings, so we read them verbatim (unlike the
// `--sentinel-*` HSL triplets that need wrapping). Called inside a client
// effect, re-keyed on the resolved theme, so switching themes repaints.
function readVar(el: HTMLElement, name: string, fallback: string): string {
  const v = getComputedStyle(el).getPropertyValue(name).trim();
  return v || fallback;
}
function palette() {
  const el = document.documentElement;
  // Structural chrome + candles from the reference design tokens (mission
  // UI-2b): grid = --line, scale border = --line-2, axis + crosshair = --faint,
  // candles = --bull/--bear, current-price line = --acc. Fond = --panel via the
  // `.chartbox` container (the canvas layout stays transparent).
  return {
    axisText: readVar(el, '--faint', '#59617a'),
    grid: readVar(el, '--line', 'rgba(255, 255, 255, 0.07)'),
    scaleBorder: readVar(el, '--line-2', 'rgba(255, 255, 255, 0.11)'),
    crosshair: readVar(el, '--faint', '#59617a'),
    crosshairLabel: readVar(el, '--panel-3', '#182238'),
    candleBull: readVar(el, '--bull', CANDLE.bull),
    candleBear: readVar(el, '--bear', CANDLE.bear),
    priceLine: readVar(el, '--acc', '#4d9de0'),
  };
}

const MONO_FONT = "'ui-monospace', 'SFMono-Regular', 'Menlo', monospace";

export function ReadingChart({
  candles,
  structure,
  instrument,
  timeframe = null,
  livePrice = null,
  liveTs = null,
  marketClosed = false,
  marketStatusState = null,
  layers = DEFAULT_CHART_VIEW.layers,
  filter = DEFAULT_CHART_VIEW.filter,
  focus = null,
  highlightZoneId = null,
  referenceLevel = null,
  onClearHighlight,
  hiddenZoneIds = DEFAULT_CHART_VIEW.hiddenZoneIds,
  isolatedZoneIds = DEFAULT_CHART_VIEW.isolatedZoneIds,
  className,
  heightClassName = 'h-[clamp(300px,52svh,560px)]',
}: ReadingChartProps) {
  const t = useTranslations('app');
  const { resolvedTheme } = useTheme();

  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const chartRef = React.useRef<IChartApi | null>(null);
  const seriesRef = React.useRef<ISeriesApi<'Candlestick'> | null>(null);
  const markersRef = React.useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  // True once the chart has done its one-and-only auto-fit (initial load). After
  // that, data updates PRESERVE the user's zoom/pan — only the explicit "Ajuster"
  // button (or a chart recreation) refits. Reset when the chart is recreated.
  const didInitialFitRef = React.useRef(false);
  // Accumulated OHLC of the live FORMING candle (provisional, intra-bar).
  const formingRef = React.useRef<{
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
  } | null>(null);

  // The SMC overlay (OB/FVG boxes, live FVG fronts, liquidity segments) is drawn
  // by a lightweight-charts SERIES PRIMITIVE on the chart's own canvas — see
  // zoneOverlayPrimitive.ts. The library paints it inside its own render pass with
  // the SAME coordinate transform as the candles, so the boxes stay glued to the
  // axes during pan / zoom / price-drag / resize with ZERO desync. (The old HTML
  // overlay recomputed in a rAF loop and committed a frame late → the boxes
  // "vibrated" against the candles.)
  const overlayRef = React.useRef<ZoneOverlayPrimitive | null>(null);
  // Previous traced price — so we move the view only when the trace actually
  // CHANGES (set / changed / cleared), never on every data refresh (RG-1d).
  const prevRefPriceRef = React.useRef<number | null>(null);
  // Hover tooltip — descriptive text for the box / pocket under the crosshair,
  // driven by the chart's crosshair-move event. One element, updated only on
  // hover, so it never re-renders per frame.
  const [tooltip, setTooltip] = React.useState<{
    text: string;
    x: number;
    y: number;
  } | null>(null);

  // "Poches intactes seulement" — a reversible DISPLAY filter over the detected
  // pools (hides swept + broken segments, deletes nothing; the Structure panel
  // still lists every state). Persisted per browser, default = everything shown.
  const [liquidityIntactOnly, setLiquidityIntactOnly] = React.useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    try {
      return window.localStorage.getItem(LIQUIDITY_INTACT_ONLY_KEY) === '1';
    } catch {
      return false;
    }
  });
  const toggleLiquidityIntactOnly = React.useCallback(() => {
    setLiquidityIntactOnly((v) => {
      const next = !v;
      try {
        window.localStorage.setItem(LIQUIDITY_INTACT_ONLY_KEY, next ? '1' : '0');
      } catch {
        // Storage unavailable (private mode) — the toggle still works in-session.
      }
      return next;
    });
  }, []);

  // External-liquidity segments (BSL/SSL), read straight from engine-emitted
  // pools — never recomputed, never projected. Hidden entirely when the
  // "liquidity" layer is toggled off, and per-pocket through the SAME id
  // masking as the OB/FVG boxes (hide_zones / isolate_zones — isolation is
  // uniform: isolating a set of ids hides every other structure, pockets
  // included). All display-only and reversible.
  const liquidityLines = React.useMemo(
    () =>
      layers.liquidity
        ? applyZoneVisibility(
            buildLiquidityLines(structure, { intactOnly: liquidityIntactOnly }),
            hiddenZoneIds,
            isolatedZoneIds,
          )
        : [],
    [
      structure,
      layers.liquidity,
      liquidityIntactOnly,
      hiddenZoneIds,
      isolatedZoneIds,
    ],
  );
  const hasLiquidityPools = (structure.liquidity_pools ?? []).length > 0;

  // PROTOTYPE — provisional intra-candle interaction overlay derived from the
  // latest tick. Recomputed from the CURRENT price only (never persisted), so a
  // price retreat reverts it cleanly. Only describes FVG fill / OB touch — never
  // detection, never BOS/CHOCH.
  const liveOverlay = React.useMemo(
    () => buildLiveOverlay(structure, livePrice),
    [structure, livePrice],
  );
  const liveActive = typeof livePrice === 'number' && Number.isFinite(livePrice);

  // Current bar = last candle: its close anchors the proximity curation, its
  // time bounds the right edge of an active (unmitigated) box.
  const lastCandle = candles.length > 0 ? candles[candles.length - 1] : null;

  // Curated zone models (localized + bounded). Tested first so active boxes
  // layer on top when rendered; consumed zones are already excluded upstream.
  // The chat-driven DISPLAY filter (active-only / min-size / proximity) and the
  // per-kind layer visibility are applied here — both HIDE detected boxes, never
  // edit a band. With the defaults this is the unchanged full set.
  const { fvg: showFvg, ob: showOb } = layers;
  const {
    activeOnly: fActiveOnly,
    proximityOnly: fProximityOnly,
    proximityPct: fProximityPct,
    minSizePct: fMinSizePct,
  } = filter;
  const zones = React.useMemo<(ZoneModel & { tested: boolean })[]>(() => {
    const models = buildZoneModels(structure).filter((z) =>
      z.kind === 'ob' ? showOb : showFvg,
    );
    const price = lastCandle?.close ?? 0;
    const filtered = filterZoneModels(models, price, {
      activeOnly: fActiveOnly,
      proximityOnly: fProximityOnly,
      proximityPct: fProximityPct,
      minSizePct: fMinSizePct,
    });
    // Per-id masking (hide_zones / isolate_zones) — a display choice over the
    // detected set; the zones are never edited, only their boxes hidden.
    const visible = applyZoneVisibility(filtered, hiddenZoneIds, isolatedZoneIds);
    const { active, tested } = curateZones(visible, price);
    return [...tested, ...active];
  }, [
    structure,
    lastCandle?.close,
    showFvg,
    showOb,
    fActiveOnly,
    fProximityOnly,
    fProximityPct,
    fMinSizePct,
    hiddenZoneIds,
    isolatedZoneIds,
  ]);

  // Latest-value refs for the canvas click / hover subscriptions (bound ONCE to
  // the chart) so they always read the CURRENT highlight + handler without the
  // subscription having to be torn down and re-bound on every prop change.
  const highlightIdRef = React.useRef(highlightZoneId);
  highlightIdRef.current = highlightZoneId;
  const onClearHighlightRef = React.useRef(onClearHighlight);
  onClearHighlightRef.current = onClearHighlight;

  // ── Create the chart ONCE (theme is applied live below, not by recreating). ─
  React.useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const p = palette();

    const chart = createChart(container, {
      // Responsive: tracks the container box (paired with the ResizeObserver
      // below so the chart never feels cramped on small / rotated screens).
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: p.axisText,
        fontFamily: MONO_FONT, // monospace → tabular axis numbers.
        fontSize: 11,
        // On-chart TradingView logo hidden for a cleaner plot. The Apache-2.0
        // licence for Lightweight Charts™ then requires the attribution to be
        // kept elsewhere: it lives in the repo NOTICE file and is published for
        // users — with a link to https://www.tradingview.com/ — in the
        // "Attributions" section of the public /methodology page.
        attributionLogo: false,
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
      // Symmetric ~10 % top/bottom margins (mission UI-2b) so the visible candles
      // fill the frame at any range instead of hugging one edge (the library
      // default is 20 % top / 10 % bottom). The zone/liquidity primitive does NOT
      // implement autoscaleInfo, so it never inflates the price-axis autoscale —
      // framing stays candle-driven.
      rightPriceScale: {
        borderColor: p.scaleBorder,
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      // The engine emits UTC; render the axis + crosshair in the reader's LOCAL
      // timezone so the clock is never ambiguous (a discreet « Heure locale »
      // chip sits at the bottom-left). Candle times are UTC epoch seconds.
      localization: {
        locale: 'fr-FR',
        timeFormatter: (t: Time) =>
          formatLocalDayHm(new Date((t as number) * 1000)),
      },
      timeScale: {
        borderColor: p.scaleBorder,
        timeVisible: true,
        secondsVisible: false,
        // Floor on candle spacing: on a narrow phone plot the default 90-bar fit
        // would squeeze candles to ~3-4px hairlines. minBarSpacing caps the
        // density (the chart shows fewer bars rather than illegible slivers).
        minBarSpacing: 4,
        tickMarkFormatter: (t: Time, tickMarkType: TickMarkType) => {
          const d = new Date((t as number) * 1000);
          if (tickMarkType === TickMarkType.Year) return String(d.getFullYear());
          if (tickMarkType === TickMarkType.Month)
            return d.toLocaleDateString('fr-FR', { month: 'short' });
          if (tickMarkType === TickMarkType.DayOfMonth)
            return d.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit' });
          return formatLocalHm(d);
        },
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
      upColor: p.candleBull,
      downColor: p.candleBear,
      // Wick + border share the body colour (no contrasting outline).
      borderUpColor: p.candleBull,
      borderDownColor: p.candleBear,
      wickUpColor: p.candleBull,
      wickDownColor: p.candleBear,
      // Current-price line: hairline, dashed, colour follows the last move.
      priceLineVisible: true,
      priceLineWidth: 1,
      priceLineStyle: LineStyle.Dashed,
      // Current-price line follows the reference accent token (--acc), matching
      // the dashed price line + axis badge in docs/design/reference-desktop.html.
      priceLineColor: p.priceLine,
      lastValueVisible: true,
      // Floor the vertical auto-scale so a flat (closed-market) window never
      // magnifies micro-candles to full height. We take the library's own fitted
      // range and, only when it is smaller than MIN_VISIBLE_RANGE_FRAC of the
      // mid price, widen it symmetrically to that minimum. A normal session's
      // range already exceeds the floor, so this is a no-op on weekdays.
      // A traced reference level is brought into view IMPERATIVELY via the price
      // scale's setVisibleRange (RG-1d) — autoScale is off while one is pinned, so
      // this provider only governs the normal candle autoscale: floor the vertical
      // span so a flat (closed-market) window never magnifies micro-candles.
      autoscaleInfoProvider: (baseImplementation: () => AutoscaleInfo | null) => {
        const res = baseImplementation();
        if (!res || !res.priceRange) return res;
        const { minValue, maxValue } = res.priceRange;
        const mid = (minValue + maxValue) / 2;
        const span = maxValue - minValue;
        const minSpan = Math.abs(mid) * MIN_VISIBLE_RANGE_FRAC;
        if (mid === 0 || span >= minSpan) return res;
        const half = minSpan / 2;
        return {
          ...res,
          priceRange: { minValue: mid - half, maxValue: mid + half },
        };
      },
    });

    chartRef.current = chart;
    seriesRef.current = series;
    // A freshly created chart (initial mount or theme change) gets ONE auto-fit
    // from the data effect, then preserves the user's view from then on.
    didInitialFitRef.current = false;
    // Markers plugin for the BOS/CHOCH break history (set in the data effect).
    markersRef.current = createSeriesMarkers(series, []);

    // SMC overlay primitive — painted on the canvas, in phase with the candles.
    const overlay = new ZoneOverlayPrimitive();
    series.attachPrimitive(overlay);
    overlayRef.current = overlay;

    // Click on the HIGHLIGHTED (blue) zone → deselect it (toggle off). The
    // primitive's hit-test surfaces the object id under the cursor via
    // hoveredObjectId; only the currently-highlighted zone is clickable, so
    // panning over any other box is never captured (same as the old behaviour).
    const onClick = (param: MouseEventParams) => {
      const hid = highlightIdRef.current;
      const clear = onClearHighlightRef.current;
      if (!hid || !clear) return;
      if (param.hoveredObjectId === `zone:${hid}`) clear();
    };
    // Hover → descriptive tooltip near the cursor (box label / pocket
    // description). Only updates React state when the text or position actually
    // changes, so mouse-move doesn't spam re-renders.
    const onMove = (param: MouseEventParams) => {
      if (!param.point) {
        setTooltip((prev) => (prev ? null : prev));
        return;
      }
      const text = overlay.tooltipFor(param.hoveredObjectId);
      const x = param.point.x;
      const y = param.point.y;
      setTooltip((prev) => {
        if (!text) return prev ? null : prev;
        if (
          prev &&
          prev.text === text &&
          Math.abs(prev.x - x) < 2 &&
          Math.abs(prev.y - y) < 2
        ) {
          return prev;
        }
        return { text, x, y };
      });
    };
    chart.subscribeClick(onClick);
    chart.subscribeCrosshairMove(onMove);

    return () => {
      chart.unsubscribeClick(onClick);
      chart.unsubscribeCrosshairMove(onMove);
      try {
        series.detachPrimitive(overlay);
      } catch {
        // series already disposed by chart.remove() below — safe to ignore.
      }
      overlayRef.current = null;
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      markersRef.current = null;
    };
    // Created once — theme changes are applied in place below (UI-19), so the
    // chart is NOT torn down and rebuilt on every light/dark toggle (which lost
    // the user's zoom/pan and flickered).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Apply theme colours live on light/dark toggle (UI-19). applyOptions keeps
  // the chart instance — and the user's zoom/pan — instead of recreating it.
  // palette() reads the current CSS-variable colours, so it returns the new theme
  // once resolvedTheme (and the html class) have flipped. ──
  React.useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const p = palette();
    chart.applyOptions({
      layout: { textColor: p.axisText },
      grid: { horzLines: { color: p.grid } },
      crosshair: {
        vertLine: { color: p.crosshair, labelBackgroundColor: p.crosshairLabel },
        horzLine: { color: p.crosshair, labelBackgroundColor: p.crosshairLabel },
      },
      rightPriceScale: { borderColor: p.scaleBorder },
      timeScale: { borderColor: p.scaleBorder },
    });
  }, [resolvedTheme]);

  // ── Push candle data + structure price lines; fit content. ──────────────────
  React.useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series) return;

    // GARDE-FOU données : drop structurally-impossible bars (0 / negative / NaN /
    // high<low) before they reach the canvas. A single corrupt cache row would
    // otherwise paint a spike and — once the price axis fits to include it —
    // squash every real candle. We reject the data error, never a real move.
    const validCandles = candles.filter(isValidBar);
    const dropped = candles.length - validCandles.length;
    if (dropped > 0) {
      console.warn(
        `[ReadingChart] dropped ${dropped} invalid candle(s) (0/negative/NaN/high<low) before render`,
      );
    }

    // Preserve the user's zoom/pan ACROSS data updates: capture the visible range
    // before replacing the series so a background refetch (a candle closing) never
    // snaps the view. Only the very first load — or an explicit "Ajuster" — fits.
    const timeScale = chart.timeScale();
    const prevRange = didInitialFitRef.current ? timeScale.getVisibleLogicalRange() : null;

    series.setData(
      validCandles.map((c) => ({
        time: c.time as UTCTimestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      })),
    );
    // A fresh closed-candle window resets any in-progress forming bar; the live
    // effect rebuilds it from the next tick (so it's never drawn over stale data).
    formingRef.current = null;

    // BOS / CHOCH break-history markers (read-only, descriptive). One arrow per
    // detected break over the window — fixes the "sous-surfaçage" where only the
    // last bar's break ever showed. Lightweight-charts v5 does NOT ignore markers
    // older than the loaded range — createSeriesMarkers clamps them onto the
    // FIRST bar (NearestRight), stacking stale labels at the left edge — so we
    // pass the first loaded candle time and let the builder drop them.
    // The "breaks" layer can be hidden on chat request → no markers, no lines.
    const firstLoadedCandle = validCandles[0];
    markersRef.current?.setMarkers(
      layers.breaks && firstLoadedCandle
        ? buildStructureMarkers(structure, firstLoadedCandle.time as number)
        : [],
    );

    // Horizontal break-level price lines (BOS / CHOCH / retest) — hairline.
    // Hidden when the "breaks" layer is toggled off (display-only).
    const priceLines = (
      layers.breaks
        ? [
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
          ]
        : []
    ).filter((l): l is { price: number; color: string; title: string } =>
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

    // External liquidity pockets (BSL/SSL): the SEGMENT itself is drawn by the
    // canvas overlay primitive (time-bounded like the OB/FVG boxes), never a
    // full-width price line. Here we only keep the price-scale PILL for INTACT
    // pockets: a price line with the line hidden (lineVisible: false) so the
    // level reads on the axis without a band crossing the whole canvas.
    // Swept/broken pockets carry their state at the frozen contact instead.
    // B-03 declutter: on a short/narrow plot the right axis stacked one price
    // pill per intact pocket, overlapping the BOS/CHOCH/retest badges. Cap the
    // axis pills to the few pockets NEAREST the current price; every level still
    // renders on-chart as a canvas segment (overlay primitive), so nothing is hidden
    // — only the redundant, colliding axis badge is dropped for far levels.
    const MAX_LIQUIDITY_PILLS = 4;
    const lastClose = validCandles.length
      ? validCandles[validCandles.length - 1]!.close
      : null;
    const intactLiquidity = liquidityLines.filter((l) => l.status === 'intact');
    const pilledLiquidity =
      lastClose == null || intactLiquidity.length <= MAX_LIQUIDITY_PILLS
        ? intactLiquidity
        : [...intactLiquidity]
            .sort(
              (a, b) =>
                Math.abs(a.price - lastClose) - Math.abs(b.price - lastClose),
            )
            .slice(0, MAX_LIQUIDITY_PILLS);
    const createdLiquidity = pilledLiquidity.map((l) =>
      series.createPriceLine({
        price: l.price,
        color: liquidityColor(l.side, l.status),
        lineWidth: 1,
        lineVisible: false,
        axisLabelVisible: true,
        title: l.title,
      }),
    );

    // Calendar-derived reference marker (day/week open, prev extreme). A FINE
    // solid accent line whose axis label carries the repère NAME + value
    // (« Haut de la veille · 4 202,03 ») — deliberately distinct from the dashed
    // grey break lines, the hidden-line liquidity pills and the OB/FVG boxes, so
    // a temporal repère (computed from the calendar) is never mistaken for
    // detected structure. One at a time; drawn only after a click.
    const refColor = palette().priceLine;
    const createdReference = referenceLevel
      ? [
          series.createPriceLine({
            price: referenceLevel.price,
            color: refColor,
            lineWidth: 1,
            lineStyle: LineStyle.Solid,
            axisLabelVisible: true,
            title: referenceLevel.label,
          }),
        ]
      : [];
    // Bring the view to a traced level. React ONLY when the trace actually changes
    // (set / changed / cleared) — not on every data refresh — so a user's manual
    // zoom is preserved between ticks (RG-1d). The vertical range is set
    // IMPERATIVELY via the price scale's setVisibleRange (reliable), NOT by toggling
    // autoScale (which the library coalesces when the value is unchanged, so the
    // view never moved — the live « il ne ramène pas » bug):
    //   · set/changed → pin a range holding the level, the current price AND the
    //     visible candles (context between the two), with a margin, and scroll the
    //     chart element into view ;
    //   · cleared (re-click) → setAutoScale(true) returns to the candle-fitted
    //     scale (« retour à l'échelle précédente »).
    const curRefPrice = referenceLevel ? referenceLevel.price : null;
    if (curRefPrice !== prevRefPriceRef.current) {
      const ps = series.priceScale();
      if (curRefPrice != null) {
        const logical = timeScale.getVisibleLogicalRange();
        const target = computeReferenceViewRange(
          curRefPrice,
          validCandles,
          logical ? Math.max(0, Math.floor(logical.from)) : 0,
          logical ? Math.min(validCandles.length - 1, Math.ceil(logical.to)) : validCandles.length - 1,
        );
        if (target) {
          ps.setAutoScale(false);
          ps.setVisibleRange(target);
        }
        containerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        ps.setAutoScale(true); // restore the candle-fitted autoscale
      }
      prevRefPriceRef.current = curRefPrice;
    }

    // Initial view ONCE; afterwards restore the pre-update view so data
    // refreshes don't reset the user's zoom/pan. The "Ajuster" button refits.
    // On first load we right-anchor to the most recent bars (not fitContent over
    // the whole history) so the chart opens on current price action, not the
    // oldest candle. Few-bar datasets fall back to fitContent.
    if (!didInitialFitRef.current) {
      const n = validCandles.length;
      if (n > DEFAULT_VISIBLE_BARS) {
        timeScale.setVisibleLogicalRange({
          from: n - DEFAULT_VISIBLE_BARS,
          to: n - 1 + INITIAL_RIGHT_PAD_BARS,
        });
      } else {
        timeScale.fitContent();
      }
      didInitialFitRef.current = true;
    } else if (prevRange) {
      timeScale.setVisibleLogicalRange(prevRange);
    }

    return () => {
      for (const line of created) series.removePriceLine(line);
      for (const line of createdLiquidity) series.removePriceLine(line);
      for (const line of createdReference) series.removePriceLine(line);
    };
  }, [candles, structure, layers.breaks, liquidityLines, referenceLevel]);

  // ── Feed the overlay primitive the current view models. ─────────────────────
  // Pure DISPLAY mapping (no detection): the curated zones + live FVG fronts +
  // liquidity segments become plain price/time models handed to the primitive,
  // which paints them on the canvas IN PHASE with the candles. This effect runs
  // only when the underlying models change; pan / zoom / resize redraws are
  // handled by the library calling the primitive every frame — no React involved,
  // so the boxes never lag the candles (the old "vibration").
  const overlayData = React.useMemo<ZoneOverlayData>(() => {
    const zoneItems = zones.map((z) => {
      const anchor = zoneRightAnchor(z);
      const inTestLive = z.kind === 'ob' && liveOverlay.obInTest.has(z.id);
      const statusText = inTestLive
        ? t('chart.inTest')
        : z.tested
          ? t('chart.touched')
          : null;
      return {
        id: z.id,
        kind: z.kind,
        high: z.high,
        low: z.low,
        // x-start = formation; x-end = mitigation point when resolved, else null
        // → the primitive extends it a little past the current bar (an active
        // zone reads just beyond the latest candle, never an infinite band).
        leftSec: z.createdSec,
        rightSec: anchor.kind === 'mitigation' ? anchor.sec : null,
        tested: z.tested,
        inTestLive,
        statusText,
        statusLive: inTestLive,
        label: z.label,
      };
    });

    // Live FVG-fill fronts share their parent box's x-span; only for a currently
    // displayed FVG zone (so a hidden / filtered box never sprouts a live front).
    const zoneById = new Map(zones.map((z) => [z.id, z]));
    const liveFronts = liveOverlay.fvgFronts.flatMap((f) => {
      const parent = zoneById.get(f.id);
      if (!parent || parent.kind !== 'fvg') return [];
      const anchor = zoneRightAnchor(parent);
      return [
        {
          id: f.id,
          high: f.high,
          low: f.low,
          leftSec: parent.createdSec,
          rightSec: anchor.kind === 'mitigation' ? anchor.sec : null,
          label: t('chart.fvgFillLive'),
        },
      ];
    });

    const liquidity = liquidityLines.map((l) => ({
      id: l.id,
      price: l.price,
      leftSec: l.createdSec,
      rightSec: l.contactSec,
      side: l.side,
      status: l.status,
      chartLabel: l.chartLabel,
      stateText:
        l.status === 'swept'
          ? t('chart.liqSwept')
          : l.status === 'broken'
            ? t('chart.liqBroken')
            : null,
      description: l.description,
    }));

    return {
      candleTimes: candles.map((c) => c.time as number),
      currentSec: lastCandle?.time ?? null,
      zones: zoneItems,
      liveFronts,
      liquidity,
      highlightId: highlightZoneId,
      // Label-backdrop legibility only: the light design gets a light pill, the
      // three dark designs a dark one (never a zone/pocket colour).
      labelBg:
        resolvedTheme === 'light'
          ? 'rgba(255, 255, 255, 0.72)'
          : 'rgba(17, 20, 32, 0.65)',
    };
  }, [
    zones,
    liveOverlay,
    liquidityLines,
    candles,
    lastCandle?.time,
    highlightZoneId,
    resolvedTheme,
    t,
  ]);

  React.useEffect(() => {
    overlayRef.current?.setData(overlayData);
  }, [overlayData]);

  // ── Live FORMING candle (TradingView-style) ─────────────────────────────────
  // Grow the current (still-open) bar with each tick: open anchored to the last
  // CLOSED candle's close for visual continuity, high/low accumulated, close =
  // live price. PROVISIONAL — it is the rightmost forming bar, never a confirmed
  // close; it resets when a real candle closes (the data effect clears it) and
  // is rebuilt from the next tick. Closed candles and structure are untouched,
  // and BOS/CHOCH are never derived from it.
  const lastClosedTime = lastCandle?.time ?? null;
  const lastClosedClose = lastCandle?.close ?? null;
  React.useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;
    if (livePrice == null || !Number.isFinite(livePrice)) return;
    if (lastClosedTime === null || lastClosedClose === null) return;
    // GARDE-FOU tick : reject obvious garbage (0 / negative / implausibly far from
    // the last close = a feed glitch) so a bad tick never explodes the forming
    // bar's range. A real large move stays well within the band and passes — we
    // reject the data error, not the volatility.
    if (!isPlausibleTick(livePrice, lastClosedClose)) {
      console.warn(
        `[ReadingChart] ignored implausible live tick ${livePrice} (ref close ${lastClosedClose})`,
      );
      return;
    }
    const tf = timeframe ? TF_SECONDS[timeframe] : undefined;
    if (!tf) return;

    // Bucket the tick to its bar slot; require it to be AFTER the last closed bar
    // (a tick stamped inside an already-closed bar — clock edge — is ignored).
    const ts = typeof liveTs === 'number' && liveTs > 0 ? liveTs : null;
    const slot = ts !== null ? Math.floor(ts / tf) * tf : lastClosedTime + tf;
    if (slot <= lastClosedTime) return;

    const f = formingRef.current;
    if (!f || f.time !== slot) {
      formingRef.current = {
        time: slot,
        open: lastClosedClose,
        high: Math.max(lastClosedClose, livePrice),
        low: Math.min(lastClosedClose, livePrice),
        close: livePrice,
      };
    } else {
      f.high = Math.max(f.high, livePrice);
      f.low = Math.min(f.low, livePrice);
      f.close = livePrice;
    }

    const c = formingRef.current!;
    try {
      series.update({
        time: c.time as UTCTimestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      });
    } catch {
      // update() throws if time < last data point — ignore (next tick recovers).
    }
    // Advance the overlay's current-bar anchor so an active zone's right edge
    // tracks the live forming bar between ticks (cheap; requests one redraw).
    overlayRef.current?.setCurrentSec(c.time);
  }, [livePrice, liveTs, lastClosedTime, lastClosedClose, timeframe]);

  // ── Chat-driven FOCUS command (one-shot, re-triggered by nonce). ────────────
  // Re-frames the visible window only — it never changes data or geometry:
  //   · zone  — bracket the (read-only) time/price span of a DETECTED zone.
  //   · price — frame the last FOCUS_PRICE_BARS bars around the current price.
  //   · fit   — fit all candles (same as the "Ajuster" control).
  // Keyed on focus?.nonce so the same command (e.g. re-focus the same zone)
  // re-runs. A zone id that no longer resolves is a graceful no-op.
  // Latest-value refs so the focus effect (which fires only on focusNonce, to
  // avoid re-framing on every data tick) still reads the CURRENT candles /
  // structure / timeframe when it runs — not values captured by a stale closure
  // if a refetch landed between the focus command and this effect (UI-04).
  const candlesRef = React.useRef(candles);
  candlesRef.current = candles;
  const structureRef = React.useRef(structure);
  structureRef.current = structure;
  const timeframeRef = React.useRef(timeframe);
  timeframeRef.current = timeframe;

  const focusNonce = focus?.nonce ?? null;
  React.useEffect(() => {
    if (!focus) return;
    const chart = chartRef.current;
    if (!chart) return;
    const ts = chart.timeScale();
    const currentCandles = candlesRef.current;
    const currentStructure = structureRef.current;
    const currentTimeframe = timeframeRef.current;
    const times = currentCandles.map((c) => c.time as number);
    const lastTime = times.length ? times[times.length - 1]! : null;
    const barSec = (currentTimeframe ? TF_SECONDS[currentTimeframe] : undefined) ?? 0;

    if (focus.kind === 'fit') {
      ts.fitContent();
      return;
    }

    if (focus.kind === 'price') {
      if (lastTime === null || barSec <= 0) {
        ts.scrollToRealTime();
        return;
      }
      ts.setVisibleRange({
        from: (lastTime - barSec * FOCUS_PRICE_BARS) as UTCTimestamp,
        to: (lastTime + barSec * 2) as UTCTimestamp,
      });
      return;
    }

    if (focus.kind === 'zone' && focus.zoneId) {
      const model = buildZoneModels(currentStructure).find((z) => z.id === focus.zoneId);
      if (!model || lastTime === null) return;
      const zoneStart = model.createdSec;
      const zoneEnd = model.mitigatedSec ?? lastTime;
      const span = Math.max(zoneEnd - zoneStart, 0);
      // Keep plenty of context around the zone so the focus doesn't slam the
      // viewport onto a tiny band (jarring for the user). At least ~24 bars of
      // breathing room on each side, or 1.5× the zone span, whichever is larger.
      const margin = Math.max(barSec * 24, span * 1.5);
      try {
        ts.setVisibleRange({
          from: (zoneStart - margin) as UTCTimestamp,
          to: (zoneEnd + margin) as UTCTimestamp,
        });
      } catch {
        // Range outside the data — graceful no-op (next command recovers).
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusNonce]);

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

  // Timezone indicator — resolved on the client only (the browser's offset), so
  // it never mismatches the server render.
  const [tzLabel, setTzLabel] = React.useState('');
  React.useEffect(() => setTzLabel(localTimeLabel()), []);

  return (
    <div className={cn('relative w-full', className)}>
      <div
        ref={containerRef}
        className={cn(heightClassName, 'w-full')}
        role="img"
        aria-label={t('chart.canvasAria', { instrument })}
      />

      {/* Discreet local-time indicator — sits ABOVE the bottom-left controls
          (bottom-14) so the two no longer overlap on a short plot (RESP-B-05). */}
      {tzLabel && (
        <span className="pointer-events-none absolute bottom-14 left-2 z-10 select-none rounded bg-background/60 px-1.5 py-0.5 text-[10px] font-medium tracking-tight text-muted-foreground/70 backdrop-blur-sm">
          {tzLabel}
        </span>
      )}

      {/* SMC overlay (OB / FVG boxes, live FVG-fill fronts, external-liquidity
          segments) is painted ON THE CANVAS by the series primitive — in phase
          with the candles, so it stays glued to the axes during pan / zoom /
          price-drag / resize (no more vibration). The only HTML here is the hover
          tooltip below, which surfaces a box / pocket's descriptive text near the
          cursor on crosshair-move (never per frame, so it never jitters). */}
      {tooltip && (
        <div
          className="pointer-events-none absolute z-20 max-w-[220px] rounded border border-border/70 bg-background/95 px-2 py-1 text-[11px] leading-snug text-foreground shadow-sm backdrop-blur-sm"
          style={{
            left: Math.min(
              Math.max(4, tooltip.x + 12),
              Math.max(4, (containerRef.current?.clientWidth ?? 240) - 232),
            ),
            top: tooltip.y + 12,
          }}
          role="tooltip"
        >
          {tooltip.text}
        </div>
      )}

      {/* Session badge (top-right). "Marché fermé" is a PRESENT FACT and takes
          precedence: when the spot market is closed we never show the live
          badge — the app must not claim to be live when it isn't. Otherwise, the
          amber "EN DIRECT · provisoire" badge appears only while a tick is
          actually driving a provisional interaction. Neither predicts anything. */}
      {marketClosed ? (
        <div
          className="pointer-events-none absolute right-2 top-2 flex items-center gap-1 rounded-full border border-border/70 bg-muted/70 px-2 py-0.5 text-[10px] font-semibold text-muted-foreground backdrop-blur-sm"
          title={t('chart.marketClosedTitle')}
          role="status"
          aria-live="polite"
        >
          <span
            className="inline-block h-1.5 w-1.5 rounded-full bg-muted-foreground/70"
            aria-hidden
          />
          {t((marketStatusState && badgeLabelKey(marketStatusState)) || 'chart.marketClosed')}
        </div>
      ) : (
        liveActive && (
          <div
            className="pointer-events-none absolute right-2 top-2 flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold"
            style={{
              color: LIVE_COLOR,
              backgroundColor: `rgba(${LIVE_RGB}, 0.12)`,
              border: `1px solid rgba(${LIVE_RGB}, 0.5)`,
            }}
            title={t('chart.liveBadgeTitle')}
            role="status"
            aria-live="polite"
          >
            <span
              className="inline-block h-1.5 w-1.5 rounded-full"
              style={{ backgroundColor: LIVE_COLOR }}
              aria-hidden
            />
            {t('chart.liveBadge')}
          </div>
        )
      )}

      {/* Sober pan/zoom controls — visually light, ≥44px tap zone on touch.
          z-10 lifts them above the lightweight-charts canvases (which carry
          their own z-index and would otherwise intercept clicks over the
          time-axis strip). */}
      <div className="absolute bottom-2 left-2 z-10 flex gap-1">
        <ChartControl label={t('chart.zoomIn')} onClick={() => zoom(0.7)}>
          <Plus className="h-4 w-4" aria-hidden />
        </ChartControl>
        <ChartControl label={t('chart.zoomOut')} onClick={() => zoom(1.4)}>
          <Minus className="h-4 w-4" aria-hidden />
        </ChartControl>
        <ChartControl label={t('chart.fit')} onClick={fit}>
          <Maximize2 className="h-4 w-4" aria-hidden />
        </ChartControl>
        {/* "Poches intactes seulement" — reversible DISPLAY filter (hides the
            swept/broken liquidity segments, deletes nothing). Only offered when
            the liquidity layer is on and the engine emitted pockets. */}
        {layers.liquidity && hasLiquidityPools && (
          <ChartControl
            label={
              liquidityIntactOnly
                ? t('chart.liqShowAll')
                : t('chart.liqShowIntactOnly')
            }
            onClick={toggleLiquidityIntactOnly}
            pressed={liquidityIntactOnly}
          >
            <Droplets className="h-4 w-4" aria-hidden />
          </ChartControl>
        )}
      </div>
    </div>
  );
}

/** A single sober chart control: hairline border, ≥44px tap target on touch. */
function ChartControl({
  label,
  onClick,
  pressed,
  children,
}: {
  label: string;
  onClick: () => void;
  /** Toggle state (aria-pressed + emphasised style); omit for plain buttons. */
  pressed?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      aria-pressed={pressed}
      className={cn(
        'flex h-11 w-11 items-center justify-center rounded-md border border-border/60',
        'bg-background/70 text-muted-foreground backdrop-blur-sm',
        'transition-colors hover:text-foreground',
        // 44px on touch (phone + tablet); only shrink to 32px on xl desktop
        // where a mouse is the pointer. (Was sm: → served 32px to iPad touch.)
        'xl:h-8 xl:w-8',
        pressed && 'border-foreground/40 text-foreground',
      )}
    >
      {children}
    </button>
  );
}
