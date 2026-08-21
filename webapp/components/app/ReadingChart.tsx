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
  type IPriceLine,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type MouseEventParams,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';
import { Loader2, RotateCw } from 'lucide-react';
import { useTheme } from 'next-themes';
import { useTranslations, useLocale } from 'next-intl';
import { cn } from '@/lib/utils';
import { formatLocalDayHm, formatLocalHm } from '@/lib/time/localTime';
import { useLocalTimeLabel } from '@/lib/time/useLocalTimeLabel';
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
  ChartSelection,
  FocusCommand,
  ReferenceLevel,
} from '@/lib/chart/viewActions';
import { DEFAULT_CHART_VIEW } from '@/lib/chart/viewActions';
import {
  animateCamera,
  frameEvent,
  frameLevel,
  frameZone,
  type CameraFrame,
} from '@/lib/chart/focusController';
import { buildStructureMarkers, findEventById } from '@/lib/chart/structureMarkers';
import { buildLiquidityLines } from '@/lib/chart/liquidityLines';
import {
  ZoneOverlayPrimitive,
  liquidityColor,
  LIVE_RGB,
  LIVE_COLOR,
  type ZoneOverlayData,
} from '@/lib/chart/zoneOverlayPrimitive';
import { isPlausibleTick, isValidBar } from '@/lib/chart/sanitize';
import { TF_SECONDS } from '@/lib/timeframes';
import { badgeLabelKey, badgeTitleKey, formatNyTimestamp } from '@/lib/market-reading/status';
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
   * MC-1 reopen instant (ISO-8601 UTC) for a closed / paused state. When present
   * the session badge shows a visible "Réouverture …" sub-line so the reason and
   * the reopen time are explained inline — never hidden behind the hover tooltip
   * (the tooltip is invisible on touch devices). Null ⇒ label only, as before.
   */
  marketReopenTs?: string | null;
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
  /**
   * VZ-1 — the single selected element (zone / event / level), or null. Drives
   * the unified animated camera framing and the per-family visual treatment. When
   * null, the chart restores the view it held before the selection gesture.
   */
  selection?: ChartSelection | null;
  /**
   * Deselect (drop the emphasis + restore the previous view). Fired when the user
   * clicks the highlighted zone on the canvas OR presses Escape. Same toggle as
   * re-clicking the element's list entry.
   */
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
  /**
   * CHART-2 — number of bars the engine analysed for THIS reading (header
   * `analysis_window_bars`). Beyond it (older, loaded on demand) the chart carries
   * NO detected structure; the chart says so honestly instead of implying a
   * structure-free zone. Null → the boundary is unknown and no notice is shown.
   */
  analysisWindowBars?: number | null;
  /**
   * CHART-2 — load the previous page of OLDER candles (dezoom to the left edge).
   * Wired to `useCandles().loadOlder`. Omitted on surfaces without paging (landing
   * hero) — the chart then simply stops at the loaded extent.
   */
  onLoadOlder?: () => void;
  /** True while a backward history page is loading (chip on the left edge). */
  isLoadingOlder?: boolean;
  /** True once the oldest available candle has been loaded (« début des données »). */
  reachedStart?: boolean;
  /** Set when the last backward page failed — surfaces a retry affordance. */
  olderError?: Error | null;
}

// Timeframe → bar length in seconds, from the single registry (TF-1). Covers all
// six units — the previous local {M15,H1,H4} map left barSec=0 on M5/D1, which
// early-returned before framing (the reported click-to-frame failure).

/** Candle bodies — muted bull / bear fallbacks (live values come from the
 *  `--bull` / `--bear` design tokens read in palette()). */
const CANDLE = { bull: '#2F9E78', bear: '#C2693E' };

/** Break-level line colours — sober, distinguishable, hairline. */
const LEVEL = {
  bos: '#8B95A7',
  choch: '#8E84B0',
  retest: '#6E84B0',
};

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
 * CHART-2 — RESPONSIVE floor on candle spacing. This is the setting that used to
 * cap the dezoom: a FIXED `minBarSpacing: 4` meant bars could never compress below
 * 4 px, so on a ~1000 px plot no more than ~250 bars (≈ 2 trading days on M15)
 * could ever fit — the "mur invisible" the user hit. We now scale the floor to the
 * plot width: wide screens get the library's own tiny floor (dense history, real
 * dezoom), narrow phones keep a larger floor so candles never become illegible
 * hairlines. Display-only — the data is untouched.
 */
function minBarSpacingFor(plotWidth: number): number {
  if (plotWidth >= 1280) return 0.5; // desktop — full dezoom, hundreds/thousands of bars
  if (plotWidth >= 768) return 1; // tablet
  return 2; // phone — avoid sub-pixel hairlines while still allowing a wide view
}

/** Logical-range distance from the left edge that triggers an on-demand history load. */
const LOAD_OLDER_THRESHOLD_BARS = 12;

/**
 * Backstop on the number of backward pages loaded to reach a selected OLD event's
 * bar before framing it. A structure event lives inside the analysis window
 * (~500 bars) and the chart already loads 400 recent bars, so one 500-bar page
 * covers it in practice; this cap only guards against paging forever toward an
 * event whose candle is genuinely unavailable (never reached `reachedStart`).
 */
const MAX_EVENT_HISTORY_PAGES = 6;

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

/** VZ-1 — respiration cycle of the selected zone (mission §C: slow, ~2.5s). */
const ZONE_PULSE_MS = 2500;

/** True when the reader asked for reduced motion — transitions off, framing
 *  instant, respiration disabled. Read at gesture time so it always reflects the
 *  current OS setting. Guarded for SSR. */
function prefersReducedMotion(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

export function ReadingChart({
  candles,
  structure,
  instrument,
  timeframe = null,
  livePrice = null,
  liveTs = null,
  marketClosed = false,
  marketStatusState = null,
  marketReopenTs = null,
  layers = DEFAULT_CHART_VIEW.layers,
  filter = DEFAULT_CHART_VIEW.filter,
  focus = null,
  highlightZoneId = null,
  referenceLevel = null,
  selection = null,
  onClearHighlight,
  hiddenZoneIds = DEFAULT_CHART_VIEW.hiddenZoneIds,
  isolatedZoneIds = DEFAULT_CHART_VIEW.isolatedZoneIds,
  className,
  heightClassName = 'h-[clamp(300px,52svh,560px)]',
  analysisWindowBars = null,
  onLoadOlder,
  isLoadingOlder = false,
  reachedStart = false,
  olderError = null,
}: ReadingChartProps) {
  const t = useTranslations('app');
  const locale = useLocale();
  const { resolvedTheme } = useTheme();
  // Reopen time for the session badge sub-line (see marketReopenTs prop).
  const marketReopenLabel = formatNyTimestamp(marketReopenTs, locale);

  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const chartRef = React.useRef<IChartApi | null>(null);
  const seriesRef = React.useRef<ISeriesApi<'Candlestick'> | null>(null);
  const markersRef = React.useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  // True once the chart has done its one-and-only auto-fit (initial load). After
  // that, data updates PRESERVE the user's zoom/pan — only the explicit "Ajuster"
  // button (or a chart recreation) refits. Reset when the chart is recreated.
  const didInitialFitRef = React.useRef(false);
  // Identity of the candle array last pushed to the series — lets the data effect
  // skip the (costly) setData + zoom-preserve dance when it re-runs for a
  // selection-only change (VZ-1), rebuilding just the cheap markers / price lines.
  const lastCandlesRef = React.useRef<Candle[] | null>(null);
  // CHART-2 — time of the FIRST loaded bar last pushed. When a backward page
  // prepends OLDER candles the logical indices all shift, so the view is restored
  // by TIME (stable across a prepend) instead of by logical range (which would
  // jump). Null until the first data push.
  const lastFirstTimeRef = React.useRef<number | null>(null);
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
  // VZ-1 — the price window the active selection wants the vertical axis to show,
  // read by the series autoscaleInfoProvider (a stable closure). Null → the base
  // candle autoscale decides Y. Generalised from RG-1c/d's single reference price.
  const framingTargetRef = React.useRef<{ min: number; max: number } | null>(null);
  // The camera view captured before the CURRENT selection gesture, restored on
  // deselect so the user returns exactly where they were ("un geste qui se défait").
  const savedFrameRef = React.useRef<CameraFrame | null>(null);
  // Cancel handle for the in-flight camera tween (a new selection never fights a
  // running one) and for the selected-zone respiration loop.
  const cancelTweenRef = React.useRef<(() => void) | null>(null);
  const pulseRafRef = React.useRef<number | null>(null);
  // Level family: which window edge the (off-screen) current price sits beyond,
  // shown as a discreet directional chevron. A POSITION, never a movement.
  const [edgeIndicator, setEdgeIndicator] = React.useState<'above' | 'below' | null>(null);
  // Honest "this element is no longer detected in the current reading" notice.
  const [selectionMissing, setSelectionMissing] = React.useState(false);
  // Hover tooltip — descriptive text for the box / pocket under the crosshair,
  // driven by the chart's crosshair-move event. One element, updated only on
  // hover, so it never re-renders per frame.
  const [tooltip, setTooltip] = React.useState<{
    text: string;
    x: number;
    y: number;
  } | null>(null);

  // CHART-2 — honest edge notices, driven by the visible-range subscription:
  //   · atStart         — the left edge shows the OLDEST available candle.
  //   · outsideAnalysis — the visible window reaches OLDER than the analysed
  //     window, where no structure is detected (never "no structure here").
  const [atStart, setAtStart] = React.useState(false);
  const [outsideAnalysis, setOutsideAnalysis] = React.useState(false);

  // Latest-value refs so the (once-bound) visible-range subscription reads the
  // CURRENT paging state / handler without being re-bound every render.
  const onLoadOlderRef = React.useRef(onLoadOlder);
  onLoadOlderRef.current = onLoadOlder;
  const reachedStartRef = React.useRef(reachedStart);
  reachedStartRef.current = reachedStart;
  const isLoadingOlderRef = React.useRef(isLoadingOlder);
  isLoadingOlderRef.current = isLoadingOlder;
  // An EVENT selection whose confirmation bar is older than the loaded candle
  // window: we defer its framing and page history backward (loadOlder) until the
  // bar is in view, then animate once. Holds the pending event id; a new gesture
  // clears it. The page counter is a backstop so a genuinely unreachable event
  // never pages history forever.
  const pendingEventRef = React.useRef<string | null>(null);
  const pendingEventPagesRef = React.useRef(0);
  const analysisWindowBarsRef = React.useRef(analysisWindowBars);
  analysisWindowBarsRef.current = analysisWindowBars;

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
            buildLiquidityLines(structure, { intactOnly: false }),
            hiddenZoneIds,
            isolatedZoneIds,
          )
        : [],
    [structure, layers.liquidity, hiddenZoneIds, isolatedZoneIds],
  );

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
        // CHART-2: RESPONSIVE spacing floor (see minBarSpacingFor). The old fixed
        // `4` capped the dezoom to ~2 days; scaling it to the plot width lets wide
        // screens dezoom across the full loaded history while narrow phones still
        // avoid hairline candles. Kept in sync by the resize effect below.
        minBarSpacing: minBarSpacingFor(container.clientWidth),
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
      autoscaleInfoProvider: (baseImplementation: () => AutoscaleInfo | null) => {
        const res = baseImplementation();
        // VZ-1 — while a selection is framed, its price WINDOW owns the vertical
        // axis: return it verbatim (the framing rules already sized the margins
        // and 25–60% occupancy). Read from a ref so this stable closure always
        // sees the latest target without re-creating the series. When no window is
        // active, fall back to the base candle fit. Either way, the min-span floor
        // below guarantees a flat (closed-market) window never magnifies micro-
        // candles — we never dezoom into indistinct candles.
        const target = framingTargetRef.current;
        let minValue: number;
        let maxValue: number;
        if (
          target &&
          Number.isFinite(target.min) &&
          Number.isFinite(target.max) &&
          target.max > target.min
        ) {
          minValue = target.min;
          maxValue = target.max;
        } else {
          if (!res || !res.priceRange) return res;
          minValue = res.priceRange.minValue;
          maxValue = res.priceRange.maxValue;
        }
        const mid = (minValue + maxValue) / 2;
        const span = maxValue - minValue;
        const minSpan = Math.abs(mid) * MIN_VISIBLE_RANGE_FRAC;
        if (mid !== 0 && span < minSpan) {
          const half = minSpan / 2;
          minValue = mid - half;
          maxValue = mid + half;
        }
        const priceRange = { minValue, maxValue };
        return res ? { ...res, priceRange } : { priceRange };
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
      // Stop any in-flight camera tween / respiration loop before teardown.
      cancelTweenRef.current?.();
      cancelTweenRef.current = null;
      if (pulseRafRef.current != null) {
        cancelAnimationFrame(pulseRafRef.current);
        pulseRafRef.current = null;
      }
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
    // Skip the data push (and its zoom-preserve dance) when only the SELECTION
    // changed — the candle array is referentially stable across a selection, so
    // this re-run just refreshes the cheap markers / price lines below (VZ-1).
    const candlesChanged = candles !== lastCandlesRef.current;
    const prevRange =
      candlesChanged && didInitialFitRef.current
        ? timeScale.getVisibleLogicalRange()
        : null;
    // CHART-2: capture the visible TIME range too. When a backward page prepends
    // older bars, restoring the logical range would jump the view (indices shift);
    // restoring the time range keeps the exact same candles on screen.
    const prevTimeRange =
      candlesChanged && didInitialFitRef.current ? timeScale.getVisibleRange() : null;
    const newFirstTime = validCandles.length > 0 ? (validCandles[0]!.time as number) : null;
    const olderPrepended =
      candlesChanged &&
      didInitialFitRef.current &&
      lastFirstTimeRef.current !== null &&
      newFirstTime !== null &&
      newFirstTime < lastFirstTimeRef.current;
    if (candlesChanged) {
      lastCandlesRef.current = candles;
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
    }

    // BOS / CHOCH break-history markers (read-only, descriptive). One arrow per
    // detected break over the window — fixes the "sous-surfaçage" where only the
    // last bar's break ever showed. Lightweight-charts v5 does NOT ignore markers
    // older than the loaded range — createSeriesMarkers clamps them onto the
    // FIRST bar (NearestRight), stacking stale labels at the left edge — so we
    // pass the first loaded candle time and let the builder drop them.
    // The "breaks" layer can be hidden on chat request → no markers, no lines.
    // A selected EVENT keeps its confirmation-candle arrow visible even when the
    // "breaks" layer is hidden (that arrow is half of the event's visual couple),
    // and the builder emphasises it. When breaks are hidden we show ONLY that one.
    const selectedEvent = selection?.family === 'event' ? selection : null;
    const firstLoadedCandle = validCandles[0];
    markersRef.current?.setMarkers(
      firstLoadedCandle && (layers.breaks || selectedEvent)
        ? buildStructureMarkers(structure, firstLoadedCandle.time as number, {
            // STR-2 defect C: emphasise by STABLE id, never (kind, time).
            selected: selectedEvent ? { id: selectedEvent.id } : null,
            onlySelected: !layers.breaks,
          })
        : [],
    );

    // Horizontal break-level price lines (BOS / CHOCH / retest) — hairline.
    // Hidden when the "breaks" layer is toggled off (display-only).
    const priceLines = (
      layers.breaks
        ? [
            structure.current_bos && {
              price: structure.current_bos.level,
              color: LEVEL.bos,
              title: 'BOS',
            },
            structure.current_choch && {
              price: structure.current_choch.level,
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

    // De-duplicate levels drawn twice (LQ-D1 §4): a BOS retest IS the return to
    // the broken BOS level, so `retest_in_progress.level` equals the BOS level —
    // two badges stacked at the same price. The product rule forbids repeating a
    // value already shown elsewhere: keep ONE line per distinct price and fold
    // any collided title into it ("BOS · Retest"), instead of a second pill.
    const dedupedPriceLines = priceLines.reduce<
      { price: number; color: string; title: string }[]
    >((acc, l) => {
      const eps = Math.max(Math.abs(l.price), 1) * 1e-6;
      const hit = acc.find((k) => Math.abs(k.price - l.price) <= eps);
      if (hit) {
        if (!hit.title.split(' · ').includes(l.title)) {
          hit.title = `${hit.title} · ${l.title}`;
        }
        return acc;
      }
      acc.push({ ...l });
      return acc;
    }, []);

    const created = dedupedPriceLines.map((l) =>
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

    // ── VZ-1 selection visuals — distinct per family (§C) ─────────────────────
    // The camera framing + scroll-into-view are handled by the dedicated
    // selection effect below (animated, uniform for the three families). Here we
    // only draw each family's DISTINCT static mark:
    //   · LEVEL reference → a fine SOLID accent line (calendar-derived).
    //   · LEVEL liquidity → a SOLID per-side line (BSL teal / SSL rose).
    //   · EVENT           → a SOLID break-colour line at the broken level,
    //     labelled with the couple (« BOS ↓ · 14:00 »); its arrow marker is drawn
    //     on the confirmation candle above.
    // (Zone selection stays a canvas band + respiration, drawn by the primitive.)
    const accent = palette().priceLine;
    const selectionLines: IPriceLine[] = [];
    if (referenceLevel) {
      selectionLines.push(
        series.createPriceLine({
          price: referenceLevel.price,
          color: accent,
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
          axisLabelVisible: true,
          title: referenceLevel.label,
        }),
      );
    }
    if (selection?.family === 'level' && selection.kind === 'liquidity') {
      selectionLines.push(
        series.createPriceLine({
          price: selection.price,
          color: liquidityColor(selection.side ?? 'bsl', 'intact'),
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
          axisLabelVisible: true,
          title: selection.label,
        }),
      );
    }
    if (selection?.family === 'event') {
      const arrow = selection.direction === 'bullish' ? '↑' : '↓';
      // VZ-1b — full date + time (« BOS ↓ · 28/07 14:00 ») so the couple reads
      // completely; the broken LEVEL shows on the adjacent price-axis label.
      const when = formatLocalDayHm(new Date(selection.atSec * 1000));
      selectionLines.push(
        series.createPriceLine({
          price: selection.level,
          color: selection.kind === 'choch' ? LEVEL.choch : LEVEL.bos,
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
          axisLabelVisible: true,
          title: `${selection.kind.toUpperCase()} ${arrow} · ${when}`,
        }),
      );
    }

    // Initial view ONCE; afterwards restore the pre-update view so data
    // refreshes don't reset the user's zoom/pan. The "Ajuster" button refits.
    // On first load we right-anchor to the most recent bars (not fitContent over
    // the whole history) so the chart opens on current price action, not the
    // oldest candle. Few-bar datasets fall back to fitContent. Only runs when the
    // candle data actually changed — a selection-only re-run must not touch zoom.
    if (candlesChanged) {
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
      } else if (olderPrepended && prevTimeRange) {
        // A backward page landed: keep the EXACT same candles on screen (restore by
        // time — the logical indices just shifted by the prepended count).
        try {
          timeScale.setVisibleRange(prevTimeRange);
        } catch {
          if (prevRange) timeScale.setVisibleLogicalRange(prevRange);
        }
      } else if (prevRange) {
        timeScale.setVisibleLogicalRange(prevRange);
      }
      lastFirstTimeRef.current = newFirstTime;
    }

    return () => {
      for (const line of created) series.removePriceLine(line);
      for (const line of createdLiquidity) series.removePriceLine(line);
      for (const line of selectionLines) series.removePriceLine(line);
    };
  }, [candles, structure, layers.breaks, liquidityLines, referenceLevel, selection]);

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
      // CHART-2 — i18n « N zones » cluster text + the top-left badge footprint that
      // labels must avoid (so the status badge never covers a zone label).
      clusterLabel: (count: number) => t('chart.zonesCluster', { count }),
      badgeReserve: marketClosed || liveActive ? { w: 172, h: 26 } : null,
    };
  }, [
    zones,
    liveOverlay,
    liquidityLines,
    candles,
    lastCandle?.time,
    highlightZoneId,
    resolvedTheme,
    marketClosed,
    liveActive,
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

  // Base vertical fit (min low / max high) of the candles inside a time window —
  // the price extent the autoscale would pick, used to capture/restore Y.
  const candleFit = React.useCallback(
    (fromSec: number, toSec: number): { min: number; max: number } | null => {
      let lo = Infinity;
      let hi = -Infinity;
      for (const c of candlesRef.current) {
        const tt = c.time as number;
        if (tt < fromSec || tt > toSec) continue;
        if (c.low < lo) lo = c.low;
        if (c.high > hi) hi = c.high;
      }
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return null;
      return { min: lo, max: hi };
    },
    [],
  );

  // Snapshot the CURRENT camera (visible time range + current price window) so a
  // selection can be undone back to exactly where the user was (mission §A.4).
  const readCurrentFrame = React.useCallback((): CameraFrame => {
    const chart = chartRef.current;
    const times = candlesRef.current.map((c) => c.time as number);
    const lastSec = times.length ? times[times.length - 1]! : 0;
    const tf = timeframeRef.current;
    const barSec = (tf ? TF_SECONDS[tf] : undefined) ?? 0;
    let from = lastSec - DEFAULT_VISIBLE_BARS * (barSec || 1);
    let to = lastSec + (barSec || 1);
    const vr = chart?.timeScale().getVisibleRange();
    if (vr && typeof vr.from === 'number' && typeof vr.to === 'number') {
      from = vr.from;
      to = vr.to;
    }
    const price = framingTargetRef.current ?? candleFit(from, to);
    return { from, to, priceMin: price?.min ?? null, priceMax: price?.max ?? null, edge: null };
  }, [candleFit]);

  // Publish an interpolated price window to the autoscale closure and force the
  // vertical axis to recompute from it (works even without an X change: the
  // reduced-motion instant path and the final settle frame).
  const setPriceTarget = React.useCallback((range: { min: number; max: number } | null) => {
    framingTargetRef.current = range;
    seriesRef.current?.priceScale().setAutoScale(true);
  }, []);

  // Shared camera X-axis applier — used by BOTH the unified selection tween and the
  // deferred old-event framing, so the two paths move the camera identically.
  const setCameraVisibleRange = React.useCallback((r: { from: number; to: number }) => {
    try {
      chartRef.current
        ?.timeScale()
        .setVisibleRange({ from: r.from as UTCTimestamp, to: r.to as UTCTimestamp });
    } catch {
      // Range momentarily outside the data — the next frame recovers.
    }
  }, []);

  // Selected-zone respiration (§C): a slow, discreet opacity breath (~2.5s), never
  // blinking. Disabled under prefers-reduced-motion.
  const stopPulse = React.useCallback(() => {
    if (pulseRafRef.current != null) {
      cancelAnimationFrame(pulseRafRef.current);
      pulseRafRef.current = null;
    }
    overlayRef.current?.setHighlightPulse(1);
  }, []);
  const startPulse = React.useCallback(() => {
    const t0 = performance.now();
    const loop = () => {
      const phase = ((performance.now() - t0) % ZONE_PULSE_MS) / ZONE_PULSE_MS;
      // 0.7 → 1.0 opacity multiplier, sine-smooth (attracts the eye, never flashes).
      overlayRef.current?.setHighlightPulse(0.85 + 0.15 * Math.sin(phase * Math.PI * 2));
      pulseRafRef.current = requestAnimationFrame(loop);
    };
    pulseRafRef.current = requestAnimationFrame(loop);
  }, []);

  // ── Chat-driven camera command (focus_price / fit_chart), one-shot by nonce. ─
  // Zone framing now travels through the unified SELECTION effect below; this only
  // serves the chatbot's price/fit camera verbs. Re-frames the window only.
  const focusNonce = focus?.nonce ?? null;
  React.useEffect(() => {
    if (!focus) return;
    const chart = chartRef.current;
    if (!chart) return;
    const ts = chart.timeScale();
    const times = candlesRef.current.map((c) => c.time as number);
    const lastTime = times.length ? times[times.length - 1]! : null;
    const barSec = (timeframeRef.current ? TF_SECONDS[timeframeRef.current] : undefined) ?? 0;

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
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusNonce]);

  // ── VZ-1 — unified SELECTION camera (zone / event / level). ──────────────────
  // ONE gesture for the three families: bring the chart into view (§A.1), then
  // animate the camera onto the element (§A.2, ~400ms soft ease) — both axes
  // together, driven by a rAF tween (lightweight-charts has no native easing).
  // Re-click / Escape → selection null → restore the pre-selection view (§A.4).
  // The move is a CAMERA panning, never the price moving (§D). Keyed on the
  // selection identity; a missing target says so honestly and frames nothing.
  const selectionKeyStr = selection ? `${selection.family}:${selection.id}` : null;
  React.useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series) return;

    // Never let two tweens (or a stale respiration) fight the new gesture.
    cancelTweenRef.current?.();
    cancelTweenRef.current = null;
    stopPulse();
    setSelectionMissing(false);
    setEdgeIndicator(null);
    // A new gesture cancels any old-event history paging still pending.
    pendingEventRef.current = null;
    pendingEventPagesRef.current = 0;

    const reduced = prefersReducedMotion();
    const cameraChart = { setVisibleRange: setCameraVisibleRange };

    // DESELECT → restore the camera captured before the selection chain, then hand
    // the vertical axis back to the base candle autoscale.
    if (!selection) {
      const saved = savedFrameRef.current;
      savedFrameRef.current = null;
      if (!saved) {
        setPriceTarget(null);
        return;
      }
      cancelTweenRef.current = animateCamera({
        chart: cameraChart,
        from: readCurrentFrame(),
        to: saved,
        setPriceTarget,
        reducedMotion: reduced,
        onDone: () => setPriceTarget(null),
      });
      return;
    }

    const cs = candlesRef.current;
    const times = cs.map((c) => c.time as number);
    const lastSec = times.length ? times[times.length - 1]! : null;
    const barSec = (timeframeRef.current ? TF_SECONDS[timeframeRef.current] : undefined) ?? 0;
    if (lastSec === null || barSec <= 0) return;

    let frame: CameraFrame | null = null;
    if (selection.family === 'zone') {
      const model = buildZoneModels(structureRef.current).find((z) => z.id === selection.id);
      if (!model) {
        setSelectionMissing(true); // cible disparue → message honnête, aucun cadrage.
        return;
      }
      frame = frameZone({
        startSec: model.createdSec,
        lastSec,
        barSec,
        bandLow: model.low,
        bandHigh: model.high,
        // Keep the current price in view alongside the zone when the gap allows.
        price: cs.length ? cs[cs.length - 1]!.close : null,
      });
    } else if (selection.family === 'event') {
      // STR-2 defect C: resolve the selection by its STABLE id and REJECT an
      // unknown/stale id — never fall back to « the nearest timestamp ». Framing
      // is anchored to the id's own event, so a BOS is never framed onto a CHOCH.
      const resolved = findEventById(structure, selection.id);
      if (!resolved) return;
      const evSec = Math.floor(Date.parse(resolved.event.broken_at) / 1000);
      const firstSec = cs.length ? (cs[0]!.time as number) : null;
      // OLD event OUTSIDE the loaded window: its confirmation bar isn't loaded yet.
      // Rather than frame onto empty space to the left of the data, DEFER — page
      // history backward (loadOlder) until the bar is in view, then the pending-
      // event effect animates once. The « chargement de l'historique » chip
      // (isLoadingOlder) already gives honest feedback while it pages.
      if (
        firstSec !== null &&
        evSec < firstSec &&
        onLoadOlderRef.current &&
        !reachedStartRef.current
      ) {
        // Capture the restore point BEFORE any paging, and bring the chart into
        // view now so the reader sees where the framing will land (§A.1).
        if (!savedFrameRef.current) savedFrameRef.current = readCurrentFrame();
        containerRef.current?.scrollIntoView({
          behavior: reduced ? 'auto' : 'smooth',
          block: 'nearest',
        });
        pendingEventRef.current = selection.id;
        pendingEventPagesRef.current = 0;
        if (!isLoadingOlderRef.current) {
          pendingEventPagesRef.current = 1;
          onLoadOlderRef.current();
        }
        return;
      }
      const cand = cs.find((c) => Math.abs((c.time as number) - evSec) < barSec);
      frame = frameEvent({
        atSec: evSec,
        lastSec,
        barSec,
        level: resolved.event.level,
        candleLow: cand?.low,
        candleHigh: cand?.high,
      });
    } else {
      const currentPrice = cs.length ? cs[cs.length - 1]!.close : null;
      if (currentPrice === null) return;
      let ctxLo = Infinity;
      let ctxHi = -Infinity;
      for (const c of cs.slice(-48)) {
        if (c.low < ctxLo) ctxLo = c.low;
        if (c.high > ctxHi) ctxHi = c.high;
      }
      if (!Number.isFinite(ctxLo) || !Number.isFinite(ctxHi)) {
        ctxLo = currentPrice;
        ctxHi = currentPrice;
      }
      frame = frameLevel({
        level: selection.price,
        price: currentPrice,
        lastSec,
        barSec,
        context: { low: ctxLo, high: ctxHi },
      });
    }
    if (!frame) return;

    // Remember where the user was BEFORE the selection chain began (restore point).
    if (!savedFrameRef.current) savedFrameRef.current = readCurrentFrame();

    // §A.1 — bring the chart into view (matters most under 768px where it is often
    // off-screen). Smooth unless the reader asked for reduced motion.
    containerRef.current?.scrollIntoView({
      behavior: reduced ? 'auto' : 'smooth',
      block: 'nearest',
    });

    setEdgeIndicator(frame.edge ?? null);
    cancelTweenRef.current = animateCamera({
      chart: cameraChart,
      from: readCurrentFrame(),
      to: frame,
      setPriceTarget,
      reducedMotion: reduced,
    });

    if (selection.family === 'zone' && !reduced) startPulse();

    return () => {
      cancelTweenRef.current?.();
      cancelTweenRef.current = null;
      stopPulse();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectionKeyStr]);

  // ── Deferred OLD-event framing — page history in, THEN animate once. ─────────
  // Set up by the selection effect when the chosen event's bar is older than the
  // loaded window. Re-runs on every backward page (candles), on `reachedStart`,
  // and when a page finishes (isLoadingOlder): it keeps paging one page at a time
  // until the event's confirmation bar is loaded — or the true start of history /
  // the page cap is reached — then frames it exactly like an in-window event. It
  // never animates onto empty space; if the bar is genuinely unavailable it frames
  // the oldest reachable window (with the broken-level line still drawn). A new
  // selection, deselect, or Escape clears `pendingEventRef`, so this no-ops.
  React.useEffect(() => {
    const pendingId = pendingEventRef.current;
    if (!pendingId) return;
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series) return;
    const resolved = findEventById(structureRef.current, pendingId);
    if (!resolved) {
      pendingEventRef.current = null;
      pendingEventPagesRef.current = 0;
      return;
    }
    const evSec = Math.floor(Date.parse(resolved.event.broken_at) / 1000);
    const cs = candlesRef.current;
    const firstSec = cs.length ? (cs[0]!.time as number) : null;
    const covered = firstSec !== null && evSec >= firstSec;

    if (
      !covered &&
      !reachedStartRef.current &&
      pendingEventPagesRef.current < MAX_EVENT_HISTORY_PAGES
    ) {
      // Not far enough back yet → load the next page (once the current one lands).
      if (!isLoadingOlderRef.current && onLoadOlderRef.current) {
        pendingEventPagesRef.current += 1;
        onLoadOlderRef.current();
      }
      return;
    }

    // Covered, or no more history / cap reached → frame now (best-effort) & clear.
    pendingEventRef.current = null;
    pendingEventPagesRef.current = 0;
    const lastSec = cs.length ? (cs[cs.length - 1]!.time as number) : null;
    const barSec = (timeframeRef.current ? TF_SECONDS[timeframeRef.current] : undefined) ?? 0;
    if (lastSec === null || barSec <= 0) return;
    const cand = cs.find((c) => Math.abs((c.time as number) - evSec) < barSec);
    const frame = frameEvent({
      atSec: evSec,
      lastSec,
      barSec,
      level: resolved.event.level,
      candleLow: cand?.low,
      candleHigh: cand?.high,
    });
    const reduced = prefersReducedMotion();
    cancelTweenRef.current?.();
    cancelTweenRef.current = animateCamera({
      chart: { setVisibleRange: setCameraVisibleRange },
      from: readCurrentFrame(),
      to: frame,
      setPriceTarget,
      reducedMotion: reduced,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candles, reachedStart, isLoadingOlder]);

  // ── Discreet zoom / pan controls (mouse + keyboard + ≥44px touch targets). ──
  const zoom = React.useCallback((factor: number) => {
    const ts = chartRef.current?.timeScale();
    if (!ts) return;
    const range = ts.getVisibleLogicalRange();
    if (!range) return;
    const center = (range.from + range.to) / 2;
    const half = ((range.to - range.from) / 2) * factor;
    ts.setVisibleLogicalRange({ from: center - half, to: center + half });
  }, []);

  // Pan the visible window by a fraction of its span (keyboard arrows).
  const panBy = React.useCallback((fraction: number) => {
    const ts = chartRef.current?.timeScale();
    if (!ts) return;
    const range = ts.getVisibleLogicalRange();
    if (!range) return;
    const shift = (range.to - range.from) * fraction;
    ts.setVisibleLogicalRange({ from: range.from + shift, to: range.to + shift });
  }, []);

  // Jump back to the most recent candle (mission §2 — « retour rapide à la bougie
  // la plus récente »).
  const scrollToRecent = React.useCallback(() => {
    chartRef.current?.timeScale().scrollToRealTime();
  }, []);

  // Return to the DEFAULT view — the recent-bars right-anchored window the chart
  // opens on (not fitContent over the whole deep history).
  const resetDefaultView = React.useCallback(() => {
    const ts = chartRef.current?.timeScale();
    if (!ts) return;
    const n = candlesRef.current.length;
    if (n > DEFAULT_VISIBLE_BARS) {
      ts.setVisibleLogicalRange({
        from: n - DEFAULT_VISIBLE_BARS,
        to: n - 1 + INITIAL_RIGHT_PAD_BARS,
      });
    } else {
      ts.fitContent();
    }
  }, []);

  // Keyboard access (mission §3 — non-negotiable): every zoom/pan action reachable
  // without a mouse, on the focusable chart region.
  const onChartKeyDown = React.useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      switch (e.key) {
        case '+':
        case '=':
          zoom(0.7);
          break;
        case '-':
        case '_':
          zoom(1.4);
          break;
        case 'ArrowLeft':
          panBy(-0.2);
          break;
        case 'ArrowRight':
          panBy(0.2);
          break;
        case 'Home':
        case 'End':
          scrollToRecent();
          break;
        case '0':
          resetDefaultView();
          break;
        default:
          return;
      }
      e.preventDefault();
    },
    [zoom, panBy, scrollToRecent, resetDefaultView],
  );

  // CHART-2 — keep the spacing floor in sync with the plot width (responsive
  // dezoom). autoSize handles the canvas; this only re-applies minBarSpacing.
  React.useEffect(() => {
    const el = containerRef.current;
    const chart = chartRef.current;
    if (!el || !chart || typeof ResizeObserver === 'undefined') return;
    const apply = () =>
      chart.timeScale().applyOptions({ minBarSpacing: minBarSpacingFor(el.clientWidth) });
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // CHART-2 — one visible-range subscription drives the honest edge notices and the
  // on-demand history load, reading the latest paging state from refs (bound once).
  React.useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const ts = chart.timeScale();
    const onRange = (range: { from: number; to: number } | null) => {
      if (!range) return;
      const total = candlesRef.current.length;
      // Load the previous page when the left edge nears the oldest loaded bar.
      if (
        range.from < LOAD_OLDER_THRESHOLD_BARS &&
        !reachedStartRef.current &&
        !isLoadingOlderRef.current
      ) {
        onLoadOlderRef.current?.();
      }
      // « Début des données » — the oldest available candle is on screen.
      const atStartNow = reachedStartRef.current && range.from <= 1;
      setAtStart((prev) => (prev === atStartNow ? prev : atStartNow));
      // « Hors fenêtre d'analyse » — visible bars older than the analysed window.
      const awb = analysisWindowBarsRef.current;
      const outsideNow =
        awb != null && awb > 0 && total > awb && range.from < total - awb;
      setOutsideAnalysis((prev) => (prev === outsideNow ? prev : outsideNow));
    };
    ts.subscribeVisibleLogicalRangeChange(onRange);
    return () => ts.unsubscribeVisibleLogicalRangeChange(onRange);
  }, []);

  // Timezone indicator — resolved on the client only (the browser's offset) and
  // kept in sync with mid-session timezone changes. See useLocalTimeLabel.
  const tzLabel = useLocalTimeLabel();

  return (
    <div
      className={cn(
        'group relative w-full rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-ring',
        className,
      )}
      // CHART-2 — the whole plot is a focusable region so every zoom/pan action is
      // reachable from the keyboard (mission §3, non-negotiable). Shortcuts: +/−
      // zoom, ←/→ pan, Home most-recent, 0 default view.
      tabIndex={0}
      role="application"
      aria-label={t('chart.keyboardRegionAria', { instrument })}
      onKeyDown={onChartKeyDown}
    >
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

      {/* VZ-1 — honest "target gone" notice: the selected element is no longer
          detected in the current reading. No framing, nothing invented. */}
      {selectionMissing && (
        <div
          className="pointer-events-none absolute left-1/2 top-3 z-20 -translate-x-1/2 rounded-md border border-border/70 bg-background/95 px-3 py-1.5 text-[11px] font-medium text-muted-foreground shadow-sm backdrop-blur-sm"
          role="status"
          aria-live="polite"
        >
          {t('chart.selectionMissing')}
        </div>
      )}

      {/* VZ-1 — level family: when the current price is off-screen (the level is
          too far to frame both legibly) a discreet chevron marks the edge it sits
          beyond. It indicates a POSITION, never a movement (§D). */}
      {edgeIndicator && (
        <div
          className={cn(
            'pointer-events-none absolute left-1/2 z-20 -translate-x-1/2 flex items-center gap-1 rounded-full border border-border/70 bg-background/90 px-2 py-0.5 text-[10px] font-semibold text-muted-foreground backdrop-blur-sm',
            edgeIndicator === 'above' ? 'top-2' : 'bottom-14',
          )}
          role="status"
          aria-label={t(
            edgeIndicator === 'above' ? 'chart.priceAbove' : 'chart.priceBelow',
          )}
        >
          <span aria-hidden>{edgeIndicator === 'above' ? '▲' : '▼'}</span>
          {t('chart.currentPrice')}
        </div>
      )}

      {/* Session badge (top-LEFT — kept clear of the right price scale so it
          never covers an axis price label, LQ-D1 §4). "Marché fermé" is a
          PRESENT FACT and takes precedence: when the spot market is closed we
          never show the live badge — the app must not claim to be live when it
          isn't. Otherwise, the amber "EN DIRECT · provisoire" badge appears only
          while a tick is actually driving a provisional interaction. Neither
          predicts anything. */}
      {marketClosed ? (
        <div
          className="pointer-events-none absolute left-2 top-2 flex items-start gap-1.5 rounded-2xl border border-border/70 bg-muted/70 px-2 py-1 text-[10px] font-semibold text-muted-foreground backdrop-blur-sm"
          title={t((marketStatusState && badgeTitleKey(marketStatusState)) || 'chart.marketClosedTitle')}
          role="status"
          aria-live="polite"
        >
          <span
            className="mt-1 inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-muted-foreground/70"
            aria-hidden
          />
          <span className="flex flex-col leading-tight">
            <span>
              {t((marketStatusState && badgeLabelKey(marketStatusState)) || 'chart.marketClosed')}
            </span>
            {marketReopenLabel && (
              <span className="font-normal text-muted-foreground/80">
                {t('chart.reopensAt', { when: marketReopenLabel })}
              </span>
            )}
          </span>
        </div>
      ) : (
        liveActive && (
          <div
            className="pointer-events-none absolute left-2 top-2 flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold"
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

      {/* CHART-2 — « hors fenêtre d'analyse » banner: the visible window reaches
          older than the analysed bars, where NO structure is detected. Says so
          honestly so an un-annotated stretch never reads as "no structure here"
          (an absence of data presented as a result). */}
      {outsideAnalysis && (
        <div
          className="pointer-events-none absolute left-1/2 top-3 z-20 -translate-x-1/2 rounded-md border border-border/70 bg-background/95 px-3 py-1.5 text-center text-[11px] font-medium text-muted-foreground shadow-sm backdrop-blur-sm"
          role="status"
          aria-live="polite"
        >
          {t('chart.outsideAnalysis')}
        </div>
      )}

      {/* CHART-2 — left-edge history notices. One at a time: loading a backward
          page › page failed (retry) › start of available history reached. The
          already-drawn candles NEVER disappear during a load (they stay on the
          canvas); this is only the state of the not-yet-loaded left side. */}
      <div className="absolute left-2 top-1/2 z-20 flex -translate-y-1/2 flex-col gap-1">
        {isLoadingOlder ? (
          <span
            className="pointer-events-none flex items-center gap-1.5 rounded-md border border-border/70 bg-background/90 px-2 py-1 text-[11px] font-medium text-muted-foreground shadow-sm backdrop-blur-sm"
            role="status"
            aria-live="polite"
          >
            <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
            {t('chart.loadingHistory')}
          </span>
        ) : olderError ? (
          <span
            className="flex items-center gap-1.5 rounded-md border border-border/70 bg-background/95 px-2 py-1 text-[11px] font-medium text-muted-foreground shadow-sm backdrop-blur-sm"
            role="status"
            aria-live="polite"
          >
            {t('chart.historyError')}
            {onLoadOlder && (
              <button
                type="button"
                onClick={onLoadOlder}
                className="inline-flex items-center gap-1 rounded border border-border/60 px-1.5 py-0.5 text-foreground transition-colors hover:bg-muted/60"
              >
                <RotateCw className="h-3 w-3" aria-hidden />
                {t('chart.retry')}
              </button>
            )}
          </span>
        ) : atStart ? (
          <span
            className="pointer-events-none flex items-center gap-1.5 rounded-md border border-border/70 bg-background/85 px-2 py-1 text-[10px] font-medium text-muted-foreground/90 shadow-sm backdrop-blur-sm"
            role="status"
            aria-live="polite"
          >
            {t('chart.historyStart')}
          </span>
        ) : null}
      </div>
    </div>
  );
}
