/**
 * In-canvas overlay for the reading chart's SMC drawings (OB / FVG boxes, the
 * provisional live FVG-fill front and the external-liquidity segments), built as
 * a lightweight-charts v5 SERIES PRIMITIVE.
 *
 * WHY A PRIMITIVE (and not an HTML overlay):
 * The boxes used to be absolutely-positioned <div>s whose left/top/width/height
 * were recomputed from priceToCoordinate / timeToCoordinate inside a React
 * requestAnimationFrame loop and pushed through React state. During a pan / zoom
 * / price-axis drag / resize the geometry changes every frame, so React
 * re-rendered the whole overlay tree each frame and committed the DOM OUT OF
 * PHASE with lightweight-charts' own canvas paint — the boxes visibly lagged and
 * "vibrated" against the candles.
 *
 * A primitive is drawn by the library INSIDE its own paint pass, using the exact
 * same coordinate transform it just applied to the candles. The zones are
 * therefore painted in the SAME frame as everything else — glued to the price /
 * time axes, zero desync, zero per-frame React work. This is the mature fix for
 * the jitter.
 *
 * The primitive is PURELY presentational: it reads the already-curated,
 * already-filtered display models the component hands it via `setData` and paints
 * them. It never detects, recomputes, projects or mutates a structure — same
 * contract as the old overlay, just rendered on the canvas.
 */
import type {
  IChartApi,
  IPrimitivePaneRenderer,
  IPrimitivePaneView,
  ISeriesApi,
  ISeriesPrimitive,
  PrimitiveHoveredItem,
  SeriesAttachedParameter,
  Time,
} from 'lightweight-charts';
import type { LiquiditySide, LiquidityStatus } from '@/types/market-reading';

// ─── Palette (moved here from ReadingChart; the overlay owns its own colours) ──

/** Sober zone palette — one base RGB per kind; alpha encodes active vs tested. */
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

/** Fill / border alpha by state — active reads, tested recedes. */
const ZONE_ALPHA = {
  active: { fill: 0.12, border: 0.45 },
  tested: { fill: 0.05, border: 0.18 },
} as const;

/** PROVISIONAL / live accent — warm amber, distinct from the confirmed palette. */
export const LIVE_RGB = '201, 162, 39'; // #C9A227
export const LIVE_COLOR = '#C9A227';

/** View-only HIGHLIGHT accent (chat "mets en évidence") — cool sky-blue. */
const HIGHLIGHT_RGB = '79, 157, 222'; // #4F9DDE
const HIGHLIGHT_COLOR = '#4F9DDE';

/** External-liquidity segment palette (per side), keyed by status for the alpha. */
export const LIQUIDITY_RGB: Record<LiquiditySide, string> = {
  bsl: '79, 163, 199', // #4FA3C7 — cool blue-teal
  ssl: '199, 127, 163', // #C77FA3 — muted rose-violet
};
const LIQUIDITY_ALPHA: Record<LiquidityStatus, number> = {
  intact: 0.9,
  swept: 0.55,
  broken: 0.4,
};
/** Dash pattern per status: intact = solid, swept = dashed, broken = dotted. */
const LIQUIDITY_DASH: Record<LiquidityStatus, number[]> = {
  intact: [],
  swept: [5, 4],
  broken: [1, 3],
};
export function liquidityColor(side: LiquiditySide, status: LiquidityStatus): string {
  return `rgba(${LIQUIDITY_RGB[side]}, ${LIQUIDITY_ALPHA[status]})`;
}

/**
 * How far an ACTIVE zone's right edge sits PAST the current bar, in multiples of
 * the bar spacing — a little breathing room that scales with zoom, never the plot
 * edge (an infinite band) and never short of the candle.
 */
const ACTIVE_ZONE_RIGHT_PAD_BARS = 1.5;

const MONO = "'ui-monospace', 'SFMono-Regular', 'Menlo', monospace";

// ─── Display models handed in by the component (view-only, pre-curated) ────────

/** One OB / FVG box to draw, in PRICE + TIME space (never pixels). */
export interface OverlayZone {
  id: string;
  kind: 'ob' | 'fvg';
  /** Price band (engine values, untouched). */
  high: number;
  low: number;
  /** Formation time (x-start), UNIX seconds. */
  leftSec: number;
  /** Mitigation point (x-end), UNIX seconds, or null → extends past current bar. */
  rightSec: number | null;
  tested: boolean;
  inTestLive: boolean;
  /** Localized status chip text ("en test" / "touché"), or null. */
  statusText: string | null;
  /** True when the chip is the live-amber "en test", false for the slate "touché". */
  statusLive: boolean;
  /** Full label for the hover tooltip ("Order Block" / "Fair Value Gap"). */
  label: string;
}

/** Provisional live FVG-fill front, sharing its parent box's x-span. */
export interface OverlayLiveFront {
  id: string;
  high: number;
  low: number;
  leftSec: number;
  rightSec: number | null;
  /** Localized "comblement live" tag (drawn only when the box is wide enough). */
  label: string;
}

/** One external-liquidity segment (BSL/SSL), time-bounded like the boxes. */
export interface OverlayLiquidity {
  id: string;
  price: number;
  leftSec: number;
  rightSec: number | null;
  side: LiquiditySide;
  status: LiquidityStatus;
  /** Left-edge tag for intact pockets ("Liquidité achat · intacte"). */
  chartLabel: string;
  /** Right-edge state tag for swept/broken ("prise" / "cassée"), or null. */
  stateText: string | null;
  /** Full descriptive tooltip text. */
  description: string;
}

export interface ZoneOverlayData {
  /** Candle times (UNIX s) — used to snap a target time to a real data point. */
  candleTimes: number[];
  /**
   * Current bar time (UNIX s): the live forming bar when a tick streams, else the
   * last closed bar. Bounds the right edge of an active box / intact pocket.
   */
  currentSec: number | null;
  zones: OverlayZone[];
  liveFronts: OverlayLiveFront[];
  liquidity: OverlayLiquidity[];
  highlightId: string | null;
  /**
   * Backdrop colour for the small in-box text labels (a semi-transparent pill),
   * resolved from the active theme by the caller so the code stays legible on any
   * of the four designs. Legibility chrome only — never a zone/pocket colour.
   */
  labelBg: string;
}

// ─── Computed pixel geometry (per frame, in phase with the canvas) ─────────────

interface BoxGeom {
  id: string;
  kind: 'ob' | 'fvg';
  left: number;
  top: number;
  width: number;
  height: number;
  tested: boolean;
  inTestLive: boolean;
  highlighted: boolean;
  statusText: string | null;
  statusLive: boolean;
}

interface FrontGeom {
  left: number;
  top: number;
  width: number;
  height: number;
  label: string;
}

interface LiqGeom {
  left: number;
  width: number;
  y: number;
  side: LiquiditySide;
  status: LiquidityStatus;
  chartLabel: string;
  stateText: string | null;
}

interface HitArea {
  externalId: string;
  left: number;
  top: number;
  right: number;
  bottom: number;
  cursor?: string;
}

// ─── Small canvas helpers ──────────────────────────────────────────────────────

/** #RRGGBB → rgba() with the given alpha. */
function hexA(hex: string, a: number): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

function roundRectPath(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/** Draw a tiny text pill (label chrome) and return its total advance width. */
function drawPill(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  text: string,
  fontPx: number,
  textColor: string,
  bg: string | null,
): number {
  ctx.font = `${fontPx}px ${MONO}`;
  ctx.textBaseline = 'top';
  const padX = 3;
  const padY = 1.5;
  const w = ctx.measureText(text).width;
  const boxW = w + padX * 2;
  const boxH = fontPx + padY * 2;
  if (bg) {
    ctx.fillStyle = bg;
    roundRectPath(ctx, x, y, boxW, boxH, 2);
    ctx.fill();
  }
  ctx.fillStyle = textColor;
  ctx.fillText(text, x + padX, y + padY);
  return boxW;
}

// ─── The primitive ─────────────────────────────────────────────────────────────

/**
 * A single series primitive that owns ALL the SMC overlay drawing. Attach it once
 * to the candlestick series; feed it view models with `setData`. It caches pixel
 * geometry in `updateAllViews` (called by the library right before each paint) so
 * the renderer just blits, and it exposes `tooltipFor` so the component can show a
 * hover tooltip from the crosshair-move event's `hoveredObjectId`.
 */
export class ZoneOverlayPrimitive implements ISeriesPrimitive<Time> {
  private _chart: IChartApi | null = null;
  private _series: ISeriesApi<'Candlestick'> | null = null;
  private _requestUpdate: (() => void) | null = null;
  private _data: ZoneOverlayData | null = null;

  private _boxes: BoxGeom[] = [];
  private _fronts: FrontGeom[] = [];
  private _liqs: LiqGeom[] = [];
  private _hit: HitArea[] = [];
  private _tips = new Map<string, string>();

  private readonly _paneView: IPrimitivePaneView;
  private readonly _views: readonly IPrimitivePaneView[];

  constructor() {
    const renderer: IPrimitivePaneRenderer = {
      draw: (target) => {
        target.useMediaCoordinateSpace(({ context }) => this._draw(context));
      },
    };
    this._paneView = {
      zOrder: () => 'normal',
      renderer: () => renderer,
    };
    this._views = [this._paneView];
  }

  attached(param: SeriesAttachedParameter<Time>): void {
    this._chart = param.chart;
    this._series = param.series as ISeriesApi<'Candlestick'>;
    this._requestUpdate = param.requestUpdate;
  }

  detached(): void {
    this._chart = null;
    this._series = null;
    this._requestUpdate = null;
    this._data = null;
    this._boxes = [];
    this._fronts = [];
    this._liqs = [];
    this._hit = [];
    this._tips.clear();
  }

  paneViews(): readonly IPrimitivePaneView[] {
    return this._views;
  }

  /** Called by the library before each paint (viewport / data change) — in phase. */
  updateAllViews(): void {
    this._compute();
  }

  /** Feed new view models; triggers an in-phase redraw. */
  setData(data: ZoneOverlayData): void {
    this._data = data;
    this._compute();
    this._requestUpdate?.();
  }

  /**
   * Update ONLY the current-bar anchor (live forming bar advancing) without
   * rebuilding the whole payload — keeps an active zone's right edge tracking the
   * live bar between ticks. Cheap; requests a redraw.
   */
  setCurrentSec(sec: number | null): void {
    if (!this._data) return;
    if (this._data.currentSec === sec) return;
    this._data = { ...this._data, currentSec: sec };
    this._compute();
    this._requestUpdate?.();
  }

  /** Full descriptive text for a hovered object id, or null. */
  tooltipFor(externalId: unknown): string | null {
    if (typeof externalId !== 'string') return null;
    return this._tips.get(externalId) ?? null;
  }

  /**
   * Hit test for cursor + click/hover routing. Returns the TOPMOST drawing under
   * the cursor: the highlighted zone (pointer cursor, so its deselect affordance
   * shows) takes precedence, then zones, then liquidity bands.
   */
  hitTest(x: number, y: number): PrimitiveHoveredItem | null {
    for (let i = this._hit.length - 1; i >= 0; i -= 1) {
      const a = this._hit[i]!;
      if (x >= a.left && x <= a.right && y >= a.top && y <= a.bottom) {
        return {
          externalId: a.externalId,
          zOrder: 'normal',
          cursorStyle: a.cursor,
        } as PrimitiveHoveredItem;
      }
    }
    return null;
  }

  // ── Geometry (pure pixel mapping from the current transform) ──────────────────
  private _compute(): void {
    this._boxes = [];
    this._fronts = [];
    this._liqs = [];
    this._hit = [];
    this._tips.clear();

    const data = this._data;
    const chart = this._chart;
    const series = this._series;
    if (!data || !chart || !series) return;

    const ts = chart.timeScale();
    const times = data.candleTimes;
    const plotRight = ts.width();

    const snap = (sec: number): number | null => {
      if (!times.length) return null;
      let best = times[0]!;
      let bestD = Math.abs(best - sec);
      for (const t of times) {
        const d = Math.abs(t - sec);
        if (d < bestD) {
          best = t;
          bestD = d;
        }
      }
      return best;
    };
    const toX = (sec: number): number | null => {
      const s = snap(sec);
      if (s === null) return null;
      const c = ts.timeToCoordinate(s as Time);
      return c === null ? null : (c as number);
    };
    const toY = (price: number): number | null => {
      const c = series.priceToCoordinate(price);
      return c === null ? null : (c as number);
    };

    const barSpacing = ts.options().barSpacing ?? 6;
    const currentSec = data.currentSec;
    const activeRightX = (): number => {
      if (currentSec === null) return plotRight;
      const raw = ts.timeToCoordinate(currentSec as Time);
      if (raw === null) return plotRight;
      return (raw as number) + barSpacing * ACTIVE_ZONE_RIGHT_PAD_BARS;
    };

    // Liquidity segments first (lowest layer), then zones, then the highlighted
    // zone last — the hit-test walks from the end, so this ordering makes the
    // highlighted box the top-most clickable target.
    for (const l of data.liquidity) {
      const y = toY(l.price);
      if (y === null) continue;
      const x1 = Number.isFinite(l.leftSec) ? toX(l.leftSec) : 0;
      const x2 = l.rightSec === null ? activeRightX() : toX(l.rightSec);
      if (x1 === null || x2 === null) continue;
      const left = Math.min(x1, x2);
      const width = Math.max(2, Math.abs(x2 - x1));
      this._liqs.push({
        left,
        width,
        y,
        side: l.side,
        status: l.status,
        chartLabel: l.chartLabel,
        stateText: l.stateText,
      });
      const eid = `liq:${l.id}`;
      this._tips.set(eid, l.description);
      this._hit.push({
        externalId: eid,
        left,
        right: left + width,
        top: y - 5,
        bottom: y + 5,
      });
    }

    const highlightHits: HitArea[] = [];
    for (const z of data.zones) {
      const yHigh = toY(z.high);
      const yLow = toY(z.low);
      if (yHigh === null || yLow === null) continue;
      const top = Math.min(yHigh, yLow);
      const height = Math.max(2, Math.abs(yLow - yHigh));
      const x1 = toX(z.leftSec);
      const x2 = z.rightSec === null ? activeRightX() : toX(z.rightSec);
      if (x1 === null || x2 === null) continue;
      const left = Math.min(x1, x2);
      const width = Math.max(2, Math.abs(x2 - x1));
      const highlighted = data.highlightId != null && z.id === data.highlightId;
      this._boxes.push({
        id: z.id,
        kind: z.kind,
        left,
        top,
        width,
        height,
        tested: z.tested,
        inTestLive: z.inTestLive,
        highlighted,
        statusText: z.statusText,
        statusLive: z.statusLive,
      });
      const eid = `zone:${z.id}`;
      this._tips.set(eid, z.label);
      const hit: HitArea = {
        externalId: eid,
        left,
        right: left + width,
        top,
        bottom: top + height,
        cursor: highlighted ? 'pointer' : undefined,
      };
      if (highlighted) highlightHits.push(hit);
      else this._hit.push(hit);
    }
    // Highlighted zone(s) hit-test on top of everything.
    for (const h of highlightHits) this._hit.push(h);

    for (const f of data.liveFronts) {
      const yHigh = toY(f.high);
      const yLow = toY(f.low);
      if (yHigh === null || yLow === null) continue;
      const x1 = toX(f.leftSec);
      const x2 = f.rightSec === null ? activeRightX() : toX(f.rightSec);
      if (x1 === null || x2 === null) continue;
      const left = Math.min(x1, x2);
      const width = Math.max(2, Math.abs(x2 - x1));
      this._fronts.push({
        left,
        top: Math.min(yHigh, yLow),
        width,
        height: Math.max(2, Math.abs(yLow - yHigh)),
        label: f.label,
      });
    }
  }

  // ── Painting ──────────────────────────────────────────────────────────────────
  private _draw(ctx: CanvasRenderingContext2D): void {
    if (!this._data) return;
    const pillBg = this._data.labelBg;

    // 1) Liquidity segments (bottom layer).
    for (const l of this._liqs) {
      const color = liquidityColor(l.side, l.status);
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash(LIQUIDITY_DASH[l.status]);
      const y = Math.round(l.y) + 0.5;
      ctx.beginPath();
      ctx.moveTo(l.left, y);
      ctx.lineTo(l.left + l.width, y);
      ctx.stroke();
      ctx.setLineDash([]);
      // Contact marker at the frozen right end: dot = prise, × = cassée.
      const rx = l.left + l.width;
      if (l.status === 'swept') {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(rx, y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      } else if (l.status === 'broken') {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(rx - 3, y - 3);
        ctx.lineTo(rx + 3, y + 3);
        ctx.moveTo(rx + 3, y - 3);
        ctx.lineTo(rx - 3, y + 3);
        ctx.stroke();
      }
      ctx.restore();
    }

    // 2) Zone boxes (fill + border).
    for (const b of this._boxes) {
      const rgb = ZONE_RGB[b.kind];
      const a = b.tested ? ZONE_ALPHA.tested : ZONE_ALPHA.active;
      ctx.save();
      ctx.fillStyle = b.highlighted
        ? `rgba(${HIGHLIGHT_RGB}, ${Math.max(a.fill, 0.16)})`
        : `rgba(${rgb}, ${a.fill})`;
      ctx.fillRect(b.left, b.top, b.width, b.height);

      const bx = Math.round(b.left) + 0.5;
      const by = Math.round(b.top) + 0.5;
      const bw = Math.round(b.width);
      const bh = Math.round(b.height);
      if (b.highlighted) {
        // Cool solid ring + a faint outer glow (the old box-shadow).
        ctx.strokeStyle = hexA(HIGHLIGHT_COLOR, 0.95);
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.strokeRect(bx, by, bw, bh);
        ctx.strokeStyle = `rgba(${HIGHLIGHT_RGB}, 0.35)`;
        ctx.lineWidth = 1;
        ctx.strokeRect(bx - 1.5, by - 1.5, bw + 3, bh + 3);
      } else if (b.inTestLive) {
        ctx.strokeStyle = `rgba(${LIVE_RGB}, 0.85)`;
        ctx.lineWidth = 1;
        ctx.setLineDash([]);
        ctx.strokeRect(bx, by, bw, bh);
      } else {
        ctx.strokeStyle = `rgba(${rgb}, ${a.border})`;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(bx, by, bw, bh);
      }
      ctx.restore();
    }

    // 3) Live FVG-fill fronts (amber), drawn inside their confirmed box.
    for (const f of this._fronts) {
      ctx.save();
      ctx.fillStyle = `rgba(${LIVE_RGB}, 0.16)`;
      ctx.fillRect(f.left, f.top, f.width, f.height);
      ctx.strokeStyle = `rgba(${LIVE_RGB}, 0.8)`;
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.strokeRect(
        Math.round(f.left) + 0.5,
        Math.round(f.top) + 0.5,
        Math.round(f.width),
        Math.round(f.height),
      );
      ctx.restore();
    }

    // 4) Labels on top (so geometry never occludes them).
    for (const b of this._boxes) this._drawZoneLabel(ctx, b, pillBg);
    for (const f of this._fronts) {
      if (f.width < 60) continue;
      ctx.save();
      ctx.font = `9px ${MONO}`;
      ctx.textBaseline = 'top';
      ctx.fillStyle = LIVE_COLOR;
      const tw = ctx.measureText(f.label).width;
      ctx.fillText(f.label, f.left + f.width - tw - 2, f.top + 1);
      ctx.restore();
    }
    for (const l of this._liqs) this._drawLiquidityLabel(ctx, l);
  }

  private _drawZoneLabel(
    ctx: CanvasRenderingContext2D,
    b: BoxGeom,
    pillBg: string,
  ): void {
    const codeSize = b.tested ? 9 : 10;
    const hasStatus = b.inTestLive || b.tested;
    const narrow = b.width < (hasStatus ? 66 : 22);
    const clusterH = codeSize + 3;
    // The box's left edge is a pure time coordinate and often sits off-plot (old
    // formation candle). Slide the cluster to the box's first visible pixel so the
    // code stays readable — label chrome only, the box geometry is never clamped.
    let x = Math.max(2, b.left + (narrow ? 0 : 4));
    let y = narrow ? b.top - clusterH - 2 : b.top + 3;
    if (y < 0) y = b.top + 3; // above the plot → drop back inside

    ctx.save();
    const codeColor = ZONE_LABEL[b.kind];
    const adv = drawPill(
      ctx,
      x,
      y,
      ZONE_CODE[b.kind],
      codeSize,
      b.tested ? hexA(codeColor, 0.7) : codeColor,
      pillBg,
    );
    x += adv + 3;
    if (b.statusText) {
      const stColor = b.statusLive ? LIVE_COLOR : ZONE_LABEL[b.kind];
      const stBg = b.statusLive
        ? `rgba(${LIVE_RGB}, 0.16)`
        : `rgba(${ZONE_RGB[b.kind]}, 0.18)`;
      drawPill(ctx, x, y, b.statusText, 9, stColor, stBg);
    }
    ctx.restore();
  }

  private _drawLiquidityLabel(
    ctx: CanvasRenderingContext2D,
    l: LiqGeom,
  ): void {
    const labelColor = `rgba(${LIQUIDITY_RGB[l.side]}, 0.95)`;
    ctx.save();
    ctx.font = `9px ${MONO}`;
    ctx.textBaseline = 'bottom';
    if (l.status === 'intact') {
      ctx.fillStyle = labelColor;
      ctx.fillText(l.chartLabel, Math.max(2, l.left) + 2, l.y - 3);
    } else if (l.stateText) {
      ctx.globalAlpha = l.status === 'broken' ? 0.7 : 0.9;
      ctx.fillStyle = labelColor;
      const tw = ctx.measureText(l.stateText).width;
      ctx.fillText(l.stateText, l.left + l.width - tw - 4, l.y - 3);
    }
    ctx.restore();
  }
}
