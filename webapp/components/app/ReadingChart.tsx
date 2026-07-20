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
} from '@/lib/chart/viewActions';
import { DEFAULT_CHART_VIEW } from '@/lib/chart/viewActions';
import { buildStructureMarkers } from '@/lib/chart/structureMarkers';
import { buildLiquidityLines } from '@/lib/chart/liquidityLines';
import type { LiquiditySide, LiquidityStatus } from '@/types/market-reading';
import { isPlausibleTick, isValidBar } from '@/lib/chart/sanitize';
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
 * How far an ACTIVE zone's right edge sits PAST the current bar, in multiples of
 * the bar spacing. Small, deliberate breathing room — "a little to the right of
 * the candle" — that scales with zoom. NOT to the plot edge (would read as an
 * infinite band) and never short of / before the candle. Clamped to the plot so
 * it can't overflow the price gutter.
 */
const ACTIVE_ZONE_RIGHT_PAD_BARS = 1.5;

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

/**
 * Pixel geometry for a rendered, TIME-BOUNDED liquidity segment: x-start at the
 * pocket's formation, x-end at the current bar while intact or FROZEN at the
 * first contact once swept/broken (recomputed on scale / resize, like ZoneRect).
 */
interface LiquidityRect {
  id: string;
  left: number;
  width: number;
  /** Pixel y of the resting-liquidity level. */
  y: number;
  side: LiquiditySide;
  status: LiquidityStatus;
  /** Left-edge tag, e.g. "Liquidité achat · intacte". */
  chartLabel: string;
  /** Full descriptive tooltip text. */
  description: string;
}

/** Geometry-equal guard for the liquidity segments (same rAF pattern). */
function liquidityRectsEqual(a: LiquidityRect[], b: LiquidityRect[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i]!;
    const y = b[i]!;
    if (
      x.id !== y.id ||
      x.status !== y.status ||
      qpx(x.left) !== qpx(y.left) ||
      qpx(x.width) !== qpx(y.width) ||
      qpx(x.y) !== qpx(y.y)
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
 * External-liquidity segment palette. Per side (buy-side above / sell-side
 * below), deliberately distinct from BOTH the candle bull/bear hues (so a pool
 * never reads as a buy/sell instruction) and the slate/purple break lines. Cool
 * blue-teal = BSL, muted rose-violet = SSL. The STATUS drives the alpha + the
 * CSS border style: intact = crisp solid, swept (prise) = dashed + dimmed,
 * broken (cassée) = tight dotted + very dim. Rendu cible validé.
 */
const LIQUIDITY_RGB: Record<LiquiditySide, string> = {
  bsl: '79, 163, 199', // #4FA3C7 — cool blue-teal
  ssl: '199, 127, 163', // #C77FA3 — muted rose-violet
};
const LIQUIDITY_ALPHA: Record<LiquidityStatus, number> = {
  intact: 0.9,
  swept: 0.55,
  broken: 0.4,
};
const LIQUIDITY_BORDER_STYLE: Record<LiquidityStatus, 'solid' | 'dashed' | 'dotted'> = {
  intact: 'solid',
  swept: 'dashed',
  broken: 'dotted',
};
function liquidityColor(side: LiquiditySide, status: LiquidityStatus): string {
  return `rgba(${LIQUIDITY_RGB[side]}, ${LIQUIDITY_ALPHA[status]})`;
}

/** localStorage key for the "intact pockets only" display toggle. */
const LIQUIDITY_INTACT_ONLY_KEY = 'mia.chart.liquidityIntactOnly';

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
 * View-only HIGHLIGHT accent for a zone the chat asked to emphasise. A COOL
 * sky-blue, deliberately distinct from both the confirmed slate/blue palette and
 * the warm-amber live accent — so "mis en évidence à la demande" never reads as a
 * detection state or an intra-candle outcome. Display emphasis only.
 */
const HIGHLIGHT_RGB = '79, 157, 222'; // #4F9DDE
const HIGHLIGHT_COLOR = '#4F9DDE';

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
  timeframe = null,
  livePrice = null,
  liveTs = null,
  marketClosed = false,
  layers = DEFAULT_CHART_VIEW.layers,
  filter = DEFAULT_CHART_VIEW.filter,
  focus = null,
  highlightZoneId = null,
  onClearHighlight,
  hiddenZoneIds = DEFAULT_CHART_VIEW.hiddenZoneIds,
  isolatedZoneIds = DEFAULT_CHART_VIEW.isolatedZoneIds,
  className,
  heightClassName = 'h-[clamp(300px,52svh,560px)]',
}: ReadingChartProps) {
  const t = useTranslations('app');
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';

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

  const [zoneRects, setZoneRects] = React.useState<ZoneRect[]>([]);
  const [liveFvgRects, setLiveFvgRects] = React.useState<LiveFvgRect[]>([]);
  const [liquidityRects, setLiquidityRects] = React.useState<LiquidityRect[]>([]);
  // Width of the right price-scale gutter, tracked so the zone overlay can end
  // exactly at the plot edge. Clipping happens THERE (container `overflow-hidden`
  // + `right` inset) — box geometry itself is pure time+price and is never
  // clamped to screen coordinates (a screen-clamped edge deforms under pan).
  const [priceGutterWidth, setPriceGutterWidth] = React.useState(0);

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
      rightPriceScale: { borderColor: p.scaleBorder },
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
      // Floor the vertical auto-scale so a flat (closed-market) window never
      // magnifies micro-candles to full height. We take the library's own fitted
      // range and, only when it is smaller than MIN_VISIBLE_RANGE_FRAC of the
      // mid price, widen it symmetrically to that minimum. A normal session's
      // range already exceeds the floor, so this is a no-op on weekdays.
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
    // HTML overlay (time-bounded like the OB/FVG boxes — see liquidityRects),
    // never a full-width price line. Here we only keep the price-scale PILL for
    // INTACT pockets: a price line with the line hidden (lineVisible: false) so
    // the level reads on the axis without a band crossing the whole canvas.
    // Swept/broken pockets carry their state at the frozen contact instead.
    const createdLiquidity = liquidityLines
      .filter((l) => l.status === 'intact')
      .map((l) =>
        series.createPriceLine({
          price: l.price,
          color: liquidityColor(l.side, l.status),
          lineWidth: 1,
          lineVisible: false,
          axisLabelVisible: true,
          title: l.title,
        }),
      );

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
    };
  }, [candles, structure, layers.breaks, liquidityLines]);

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

    const candleTimes = candles.map((c) => c.time as number);
    const lastTime = candleTimes.length ? candleTimes[candleTimes.length - 1]! : null;
    // Snap a target time to the nearest candle so timeToCoordinate (which maps
    // data points) always resolves. Used for the box LEFT edge (formation) and a
    // mitigated box's RIGHT edge; an active box's right edge is the current bar
    // plus a small pad.
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

    const computeRects = (): {
      rects: ZoneRect[];
      live: LiveFvgRect[];
      liq: LiquidityRect[];
      gutter: number;
    } => {
      // Both x-edges are PURE time coordinates — never clamped to the screen.
      // A screen-clamped edge sticks to the viewport while the other follows the
      // graph, so the box deformed during a horizontal pan (the live FVG-fill
      // frame visibly stretched). Off-plot geometry is fine: the overlay
      // container clips at the plot edge (overflow-hidden + gutter inset), which
      // renders the exact same visible surface without deforming the box.
      const gutter = chart.priceScale('right').width();
      const plotRight = Math.max(0, container.clientWidth - gutter);
      const xAt = (sec: number): number | null => {
        const snapped = snapToCandle(sec);
        if (snapped === null) return null;
        return timeScale.timeToCoordinate(snapped as UTCTimestamp);
      };

      // Right edge for an ACTIVE zone: the CURRENT bar (live forming bar when a
      // tick streams, else the last closed bar — both are real series points, so
      // timeToCoordinate resolves directly without snapping) plus a small pad of
      // bar-widths. A little past the candle, scaled to zoom — never the plot
      // edge, never before the candle. Falls back to the plot edge only if the
      // current bar has no coordinate at all (not on the time scale).
      const currentSec = formingRef.current?.time ?? lastTime;
      const barSpacing = timeScale.options().barSpacing ?? 6;
      const activeRightX = (): number => {
        if (currentSec === null) return plotRight;
        const raw = timeScale.timeToCoordinate(currentSec as UTCTimestamp);
        if (raw === null) return plotRight;
        return raw + barSpacing * ACTIVE_ZONE_RIGHT_PAD_BARS;
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

        // x-start = formation. x-end depends on the zone's right anchor:
        //  · MITIGATED → its mitigation point (bounded, never over-extended).
        //  · ACTIVE → a little PAST the current bar (see activeRightX): an active
        //    zone is valid now, so it reads just beyond the latest candle without
        //    becoming an infinite band.
        const xStart = xAt(z.createdSec);
        const anchor = zoneRightAnchor(z);
        const xEnd = anchor.kind === 'mitigation' ? xAt(anchor.sec) : activeRightX();
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
      // Liquidity segments — SAME time-bounding mechanics as the zone boxes:
      // x-start = formation candle (plot left edge if unparseable — a real
      // pocket is never dropped), x-end = the current bar + pad while INTACT
      // (like an active OB) or FROZEN at the first contact (engine-emitted
      // swept_at / broken_at) once the level was touched. Read-only geometry,
      // pure time+price like the boxes — the container clips at the plot edge.
      const liq: LiquidityRect[] = [];
      for (const l of liquidityLines) {
        const y = series.priceToCoordinate(l.price);
        if (y === null) continue;
        const xStart = Number.isFinite(l.createdSec) ? xAt(l.createdSec) : 0;
        const xEnd = l.contactSec !== null ? xAt(l.contactSec) : activeRightX();
        if (xStart === null || xEnd === null) continue;
        liq.push({
          id: l.id,
          left: Math.min(xStart, xEnd),
          width: Math.max(2, Math.abs(xEnd - xStart)),
          y,
          side: l.side,
          status: l.status,
          chartLabel: l.chartLabel,
          description: l.description,
        });
      }

      return { rects, live, liq, gutter };
    };

    // Animation-frame loop: recompute in phase with the canvas, commit to React
    // only when the geometry changes (rectsEqual guard) so an idle chart costs
    // a cheap coordinate read per frame and zero re-renders.
    let prev: ZoneRect[] = [];
    let prevLive: LiveFvgRect[] = [];
    let prevLiq: LiquidityRect[] = [];
    let prevGutter = -1;
    let raf = 0;
    const tick = () => {
      // Bail if the chart was disposed/recreated (next-themes resolves the theme
      // after hydration, which re-runs the create-chart effect and remove()s this
      // one). The captured `chart`/`series` would then be disposed — calling them
      // throws "Object is disposed". Stop this stale loop; the fresh effect run
      // owns the new chart. The try/catch is a final guard against a disposal
      // that races mid-frame.
      if (chartRef.current !== chart || seriesRef.current !== series) return;
      let next: {
        rects: ZoneRect[];
        live: LiveFvgRect[];
        liq: LiquidityRect[];
        gutter: number;
      };
      try {
        next = computeRects();
      } catch {
        return; // disposed mid-frame — stop without rescheduling
      }
      if (!rectsEqual(prev, next.rects)) {
        prev = next.rects;
        setZoneRects(next.rects);
      }
      if (!liveRectsEqual(prevLive, next.live)) {
        prevLive = next.live;
        setLiveFvgRects(next.live);
      }
      if (!liquidityRectsEqual(prevLiq, next.liq)) {
        prevLiq = next.liq;
        setLiquidityRects(next.liq);
      }
      if (next.gutter !== prevGutter) {
        prevGutter = next.gutter;
        setPriceGutterWidth(next.gutter);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
    };
  }, [zones, candles, liveOverlay, liquidityLines]);

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
  }, [livePrice, liveTs, lastClosedTime, lastClosedClose, timeframe]);

  // ── Chat-driven FOCUS command (one-shot, re-triggered by nonce). ────────────
  // Re-frames the visible window only — it never changes data or geometry:
  //   · zone  — bracket the (read-only) time/price span of a DETECTED zone.
  //   · price — frame the last FOCUS_PRICE_BARS bars around the current price.
  //   · fit   — fit all candles (same as the "Ajuster" control).
  // Keyed on focus?.nonce so the same command (e.g. re-focus the same zone)
  // re-runs. A zone id that no longer resolves is a graceful no-op.
  const focusNonce = focus?.nonce ?? null;
  React.useEffect(() => {
    if (!focus) return;
    const chart = chartRef.current;
    if (!chart) return;
    const ts = chart.timeScale();
    const times = candles.map((c) => c.time as number);
    const lastTime = times.length ? times[times.length - 1]! : null;
    const barSec = (timeframe ? TF_SECONDS[timeframe] : undefined) ?? 0;

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
      const model = buildZoneModels(structure).find((z) => z.id === focus.zoneId);
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

      {/* Localized OB / FVG boxes, layered over the chart canvas. Active boxes
          read crisp; tested boxes recede (ghost fill/border). Every box carries
          a short OB / FVG type code at its top-left so the kind is always
          identifiable — crisp on active, dimmer + smaller on tested. An active
          OB the live price is inside gets a warm-amber "en test" accent (a
          PROVISIONAL, intra-candle state — distinct hue from the confirmed
          palette so it's never read as a candle-confirmed outcome).
          The container ends at the PLOT edge (right inset = price-scale gutter)
          and clips overflow: boxes are anchored purely to time+price (their
          geometry can extend off-plot) and get clipped HERE, so no edge is ever
          pinned to a screen coordinate — panning/zooming moves them rigidly. */}
      <div
        className="pointer-events-none absolute inset-y-0 left-0 overflow-hidden"
        style={{ right: priceGutterWidth }}
      >
        {zoneRects.map((r) => {
          const rgb = ZONE_RGB[r.kind];
          const a = r.tested ? ZONE_ALPHA.tested : ZONE_ALPHA.active;
          // View-only emphasis requested via chat — a cool solid ring, distinct
          // from the live-amber "en test" accent. Display only; geometry is the
          // detected band, untouched.
          const isHighlighted = highlightZoneId != null && r.id === highlightZoneId;
          const border = isHighlighted
            ? `2px solid rgba(${HIGHLIGHT_RGB}, 0.95)`
            : r.inTestLive
              ? `1px solid rgba(${LIVE_RGB}, 0.85)`
              : `1px dashed rgba(${rgb}, ${a.border})`;
          // Only the HIGHLIGHTED (blue) box is interactive — clicking it
          // deselects (toggle off). Every other box stays pointer-events:none so
          // chart panning/zooming over zones is never captured. The container is
          // pointer-events:none, so we opt this one box back in.
          const clickable = isHighlighted && onClearHighlight != null;
          return (
            <div
              key={`${r.kind}:${r.id}`}
              className="absolute"
              role={clickable ? 'button' : undefined}
              tabIndex={clickable ? 0 : undefined}
              aria-label={clickable ? t('chart.deselectZoneAria') : undefined}
              title={clickable ? t('chart.deselectZoneTitle') : undefined}
              onClick={clickable ? () => onClearHighlight() : undefined}
              onKeyDown={
                clickable
                  ? (e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        onClearHighlight();
                      }
                    }
                  : undefined
              }
              style={{
                left: r.left,
                width: r.width,
                top: r.top,
                height: r.height,
                backgroundColor: isHighlighted
                  ? `rgba(${HIGHLIGHT_RGB}, ${Math.max(a.fill, 0.16)})`
                  : `rgba(${rgb}, ${a.fill})`,
                border,
                borderRadius: 1,
                boxShadow: isHighlighted
                  ? `0 0 0 1px rgba(${HIGHLIGHT_RGB}, 0.35)`
                  : undefined,
                pointerEvents: clickable ? 'auto' : 'none',
                cursor: clickable ? 'pointer' : undefined,
              }}
            >
              {/* Type code + status grouped in ONE top-left cluster, INSIDE the
                  box with a little padding. Everything is left-anchored (the box's
                  left edge sits at the zone's creation, deep in the past) so no
                  label can reach the right price-scale gutter / live price the way
                  the old `right-1` status tags did. Narrow boxes that can't hold
                  the cluster fall back to sitting just ABOVE the box, still
                  left-aligned — never to the right. */}
              {(() => {
                // Width below which the cluster would spill past the box's right
                // edge: park it just above instead. The bare type code is tiny, so
                // only boxes carrying a status chip need the larger threshold.
                const hasStatus = r.inTestLive || r.tested;
                const narrow = r.width < (hasStatus ? 66 : 22);
                // The box's left edge is a pure time coordinate and often sits
                // OFF-plot (old formation candle). Slide the label cluster to the
                // box's first VISIBLE pixel so the type code stays readable —
                // label chrome only, the box geometry itself is never clamped.
                const labelLeft = Math.max(0, -r.left) + (narrow ? 0 : 4);
                return (
                  <div
                    className={cn(
                      'absolute flex items-center gap-1 whitespace-nowrap',
                      narrow ? 'bottom-full mb-0.5' : 'top-0.5',
                    )}
                    style={{ left: labelLeft }}
                  >
                    <span
                      className={cn(
                        'font-medium leading-none tabular-nums',
                        r.tested ? 'text-[9px] opacity-70' : 'text-[10px]',
                      )}
                      style={{ color: ZONE_LABEL[r.kind] }}
                      title={r.label}
                    >
                      {ZONE_CODE[r.kind]}
                    </span>
                    {r.inTestLive && (
                      <span
                        className="rounded-sm px-1 py-px text-[9px] font-semibold leading-none"
                        style={{
                          color: LIVE_COLOR,
                          backgroundColor: `rgba(${LIVE_RGB}, 0.16)`,
                        }}
                        title={t('chart.inTestTitle')}
                      >
                        {t('chart.inTest')}
                      </span>
                    )}
                    {/* Touched-but-alive chip: a `mitigated` OB / `partially_filled`
                        FVG that price has tapped but NOT closed through. The engine
                        still tracks it, so its box extends to the current bar (it is
                        in play) — this dim slate chip (NOT the amber live accent)
                        keeps the touched state explicit so the extension never reads
                        as untested. */}
                    {r.tested && !r.inTestLive && (
                      <span
                        className="rounded-sm px-1 py-px text-[9px] font-medium leading-none opacity-80"
                        style={{
                          color: ZONE_LABEL[r.kind],
                          backgroundColor: `rgba(${ZONE_RGB[r.kind]}, 0.18)`,
                        }}
                        title={t('chart.touchedTitle')}
                      >
                        {t('chart.touched')}
                      </span>
                    )}
                  </div>
                );
              })()}
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
            title={t('chart.fvgFillTitle')}
          >
            {/* Only label the live-fill when the box is wide enough to hold the
                text — on a narrow box the nowrap badge would overflow across
                neighbouring candles. The amber fill itself still conveys the
                state. */}
            {r.width >= 60 && (
              <span
                className="absolute right-1 top-0 whitespace-nowrap text-[9px] font-semibold leading-tight"
                style={{ color: LIVE_COLOR }}
              >
                {t('chart.fvgFillLive')}
              </span>
            )}
          </div>
        ))}

        {/* External-liquidity segments (BSL/SSL) — TIME-BOUNDED like the OB
            boxes, never full-width. INTACT: crisp solid line to the current bar,
            type tag at its left edge. SWEPT (prise): frozen at the first
            contact, dashed + dimmed, dot marker + "prise" — the pocket STAYS
            visible (touched is not broken). BROKEN (cassée): frozen, tight
            dotted + very dim, × marker + "cassée". Read-only states; labels are
            factual — no target, no direction. */}
        {liquidityRects.map((r) => {
          const color = liquidityColor(r.side, r.status);
          const labelColor = `rgba(${LIQUIDITY_RGB[r.side]}, 0.95)`;
          return (
            <div
              key={`liq:${r.id}`}
              className="absolute"
              style={{
                left: r.left,
                width: r.width,
                top: r.y,
                height: 0,
                borderTop: `1px ${LIQUIDITY_BORDER_STYLE[r.status]} ${color}`,
              }}
              title={r.description}
            >
              {r.status === 'intact' ? (
                <span
                  className="absolute whitespace-nowrap text-[9px] font-medium leading-none"
                  style={{ left: 2, top: -12, color: labelColor }}
                >
                  {r.chartLabel}
                </span>
              ) : (
                <>
                  {/* Contact marker at the FROZEN right end: dot = prise
                      (swept, still holding), × = cassée (closed through). */}
                  {r.status === 'swept' ? (
                    <span
                      className="absolute h-[5px] w-[5px] rounded-full"
                      style={{ right: -2, top: -3, backgroundColor: color }}
                      aria-hidden
                    />
                  ) : (
                    <span
                      className="absolute text-[10px] font-semibold leading-none"
                      style={{ right: -3, top: -5, color }}
                      aria-hidden
                    >
                      ×
                    </span>
                  )}
                  <span
                    className="absolute whitespace-nowrap text-[9px] font-medium leading-none"
                    style={{
                      right: 4,
                      top: -12,
                      color: labelColor,
                      opacity: r.status === 'broken' ? 0.7 : 0.9,
                    }}
                  >
                    {r.status === 'swept' ? t('chart.liqSwept') : t('chart.liqBroken')}
                  </span>
                </>
              )}
            </div>
          );
        })}
      </div>

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
          {t('chart.marketClosed')}
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
