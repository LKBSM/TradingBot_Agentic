/**
 * VZ-1 — the SINGLE focus module reused by the three selectable families
 * (zones OB/FVG · events BOS/CHOCH · levels liquidity/reference). One gesture,
 * learned once: clicking an element in a panel brings the chart into view and
 * frames the camera on that element — the user never touches the chart.
 *
 * This file is split in two:
 *   1. PURE framing math (`frameZone` / `frameEvent` / `frameLevel`) — given the
 *      element's geometry + the current candles, it returns the target camera
 *      window (a visible TIME range and a visible PRICE range). No DOM, fully
 *      unit-testable. These encode the mission's framing rules verbatim.
 *   2. A camera TWEEN runner (`animateCamera`) — lightweight-charts has NO native
 *      easing (setVisibleRange jumps), so we interpolate BOTH axes frame by frame
 *      in a requestAnimationFrame loop. The X axis is driven by
 *      `timeScale.setVisibleRange`; the Y axis by publishing an interpolated
 *      [min,max] into a ref the series `autoscaleInfoProvider` reads (generalised
 *      from RG-1c/d's single-price reference-level trace). Composed transform /
 *      opacity work only — the candles are never re-fed.
 *
 * The movement is a CAMERA panning, never the price moving: no arrow/trail from
 * the current price toward a target, no directional particle, no "importance"
 * hierarchy. If an animation could read as a market anticipation, it is excluded.
 */

import type { Time, UTCTimestamp } from 'lightweight-charts';

// ─── Framing thresholds (mission §B, validated 2026-07-27) ────────────────────

/**
 * Zone: target share of the visible height the zone band should occupy (VZ-1b —
 * relaxed from 0.42 so a click gives an immediately usable plan without a manual
 * zoom-out; ~20% ⇒ ~40% breathing room above AND below the band).
 */
export const ZONE_OCCUPANCY_TARGET = 0.2;
/** Zone: hard bounds on that share — below is illegible, above loses context. */
export const ZONE_OCCUPANCY_MIN = 0.12;
export const ZONE_OCCUPANCY_MAX = 0.3;
/** Zone: breathing room past the current price when it is folded into the view,
 *  as a fraction of the band height. */
export const ZONE_PRICE_PAD_FRAC = 0.25;
/** Event: minimum candles kept on each side of the confirmation bar. */
export const EVENT_BARS_BEFORE = 20;
export const EVENT_BARS_AFTER = 10;
/** Event/level: vertical margin as a fraction of the framed span. */
export const FRAME_MARGIN_FRAC = 0.15;
/** Level (recent bars framed horizontally around the current price). */
export const LEVEL_CONTEXT_BARS = 48;
/**
 * Level readability floor: if fitting BOTH the level and the current price would
 * squeeze the recent candle range below this share of the framed height, the
 * candles read as indistinct hairlines. We then keep a readable window on the
 * live candles and surface the off-screen level with a discreet edge indicator
 * instead of dezooming into an unreadable frame (« la lisibilité prime »).
 */
export const LEVEL_READABILITY_MIN = 0.12;

/** Camera tween duration and easing (mission §A/E: ~400ms, soft ease). */
export const CAMERA_TWEEN_MS = 400;

// ─── Types ────────────────────────────────────────────────────────────────────

/** A camera target: a visible TIME window and (optionally) a PRICE window. */
export interface CameraFrame {
  /** Visible time range, epoch seconds. */
  from: number;
  to: number;
  /**
   * Visible price range. When null the caller lets the base candle autoscale
   * decide Y (used as a safe fallback when geometry is missing).
   */
  priceMin: number | null;
  priceMax: number | null;
  /**
   * Level family only: when the current price is off-screen because the level is
   * too far to frame both legibly, this names which edge the price sits beyond —
   * a discreet chevron is drawn there. It indicates a POSITION, never a movement.
   */
  edge?: 'above' | 'below' | null;
}

/** Recent candle price extent, used to size legible windows. */
export interface PriceContext {
  /** Lowest low / highest high across the recent bars used for context. */
  low: number;
  high: number;
}

// ─── Pure framing calculators ─────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * ZONE (OB / FVG): the whole zone visible from its formation bar to the current
 * bar, the band occupying ~20% of the visible height (bounds [12%, 30%]) so ~40%
 * of breathing room sits above and below it (VZ-1b — a click gives an immediately
 * usable plan, never a frame so tight the user must zoom back out). The CURRENT
 * PRICE is kept in view alongside the band whenever the gap allows it without
 * pushing the band below the 12% floor (« voir la zone sans voir le prix n'a
 * aucune utilité »); past that floor we stop rather than crush the candles.
 */
export function frameZone(args: {
  /** Formation candle time (epoch seconds). */
  startSec: number;
  /** Current (latest) bar time (epoch seconds). */
  lastSec: number;
  /** Bar length in seconds. */
  barSec: number;
  /** Zone band price bounds. */
  bandLow: number;
  bandHigh: number;
  /** Current price — folded into the view when the gap allows (see above). */
  price?: number | null;
}): CameraFrame {
  const { startSec, lastSec, barSec, bandLow, bandHigh } = args;
  const marginX = Math.max(barSec * 3, (lastSec - startSec) * 0.08);
  const from = startSec - marginX;
  const to = lastSec + marginX;

  const bandHeight = Math.max(bandHigh - bandLow, 0);
  const center = (bandHigh + bandLow) / 2;
  // Base window: the band occupies the ~20% target (⇒ ~40% margin each side).
  // A degenerate (zero-height) band still gets a tiny window so the frame is valid.
  const height =
    bandHeight > 0 ? bandHeight / ZONE_OCCUPANCY_TARGET : Math.abs(center) * 0.01 || 1;
  let priceMin = center - height / 2;
  let priceMax = center + height / 2;

  // Fold the current price into the view when it sits outside the base window,
  // extending toward it ONLY as far as the band still occupies ≥ 12% (the
  // readability floor). Beyond that the price stays off-screen rather than
  // dezooming the candles into hairlines — the zone's legibility comes first.
  const price = args.price;
  if (price != null && Number.isFinite(price) && bandHeight > 0) {
    const maxHeight = bandHeight / ZONE_OCCUPANCY_MIN;
    const pad = bandHeight * ZONE_PRICE_PAD_FRAC;
    if (price > priceMax) {
      priceMax = Math.min(price + pad, priceMin + maxHeight);
    } else if (price < priceMin) {
      priceMin = Math.max(price - pad, priceMax - maxHeight);
    }
  }
  return { from, to, priceMin, priceMax, edge: null };
}

/**
 * EVENT (BOS / CHOCH): centred on the confirmation bar with ≥20 bars before and
 * ≥10 after (or the current bar if nearer). The broken LEVEL and the confirmation
 * candle must be visible together — that couple is what explains the event.
 */
export function frameEvent(args: {
  /** Confirmation candle time (epoch seconds). */
  atSec: number;
  /** Current (latest) bar time (epoch seconds). */
  lastSec: number;
  barSec: number;
  /** Broken level price. */
  level: number;
  /** Confirmation candle extent (falls back to the level when unavailable). */
  candleLow?: number;
  candleHigh?: number;
}): CameraFrame {
  const { atSec, lastSec, barSec, level } = args;
  const from = atSec - EVENT_BARS_BEFORE * barSec;
  // Keep ≥10 bars after, but never run past the current bar (+ a little pad).
  const to = Math.min(atSec + EVENT_BARS_AFTER * barSec, lastSec + barSec * 2);

  const cLow = args.candleLow ?? level;
  const cHigh = args.candleHigh ?? level;
  const lo = Math.min(level, cLow);
  const hi = Math.max(level, cHigh);
  const span = Math.max(hi - lo, Math.abs(level) * 0.001 || 1);
  const margin = span * FRAME_MARGIN_FRAC;
  return {
    from,
    to,
    priceMin: lo - margin,
    priceMax: hi + margin,
    edge: null,
  };
}

/**
 * LEVEL (liquidity pocket / temporal reference): the level and the current price
 * visible together with margin. If the gap is so wide the candles would be
 * crushed, keep a legible window on the live candles and flag the off-screen
 * level with an edge indicator (mission §B: « la lisibilité prime »).
 */
export function frameLevel(args: {
  /** Selected level price. */
  level: number;
  /** Current price. */
  price: number;
  /** Current (latest) bar time (epoch seconds). */
  lastSec: number;
  barSec: number;
  /** Recent candle extent (for the legible-window fallback). */
  context: PriceContext;
}): CameraFrame {
  const { level, price, lastSec, barSec, context } = args;
  const from = lastSec - LEVEL_CONTEXT_BARS * barSec;
  const to = lastSec + barSec * 2;

  const recentRange = Math.max(context.high - context.low, Math.abs(price) * 0.001 || 1);
  const lo = Math.min(level, price);
  const hi = Math.max(level, price);
  const bothSpan = (hi - lo) * (1 + 2 * FRAME_MARGIN_FRAC);
  // Would fitting both crush the recent candles below the readability floor?
  const occupancyIfBoth = recentRange / Math.max(bothSpan, 1e-9);
  if (occupancyIfBoth >= LEVEL_READABILITY_MIN) {
    const margin = (hi - lo) * FRAME_MARGIN_FRAC || recentRange * FRAME_MARGIN_FRAC;
    return { from, to, priceMin: lo - margin, priceMax: hi + margin, edge: null };
  }
  // Too far apart: keep a readable window on the live candles, indicate the level.
  const half = (recentRange * (1 + 2 * FRAME_MARGIN_FRAC)) / 2;
  const center = (context.high + context.low) / 2;
  return {
    from,
    to,
    priceMin: center - half,
    priceMax: center + half,
    // The level sits beyond the top (above) or the bottom (below) of the window.
    edge: level > center ? 'above' : 'below',
  };
}

// ─── Easing ───────────────────────────────────────────────────────────────────

/** Soft ease-in-out (mission §A: « courbe d'accélération douce »). */
export function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

// ─── Camera tween runner ──────────────────────────────────────────────────────

/** Minimal slice of the chart APIs the tween needs (keeps this testable-ish). */
export interface CameraChart {
  setVisibleRange(range: { from: number; to: number }): void;
}

export interface AnimateCameraArgs {
  chart: CameraChart;
  /** Start camera (captured before the move). */
  from: CameraFrame;
  /** Target camera. */
  to: CameraFrame;
  /**
   * Publishes the interpolated PRICE window each frame; the series
   * autoscaleInfoProvider reads it to drive the vertical axis. Passing null (only
   * at the very end of a restore) hands Y back to the base candle autoscale.
   */
  setPriceTarget(range: { min: number; max: number } | null): void;
  durationMs?: number;
  /** prefers-reduced-motion → jump straight to the target, no tween. */
  reducedMotion?: boolean;
  /** Called once the tween settles (or immediately when reduced). */
  onDone?: () => void;
  /** Monotonic clock (defaults to performance.now); injectable for tests. */
  now?: () => number;
  /** rAF scheduler (defaults to requestAnimationFrame); injectable for tests. */
  raf?: (cb: () => void) => void;
}

/**
 * Animate the camera from `from` to `to` over ~400ms with a soft ease, moving
 * BOTH axes together. Returns a cancel function (call it before starting a new
 * move so two tweens never fight). The final Y target is `to.priceMin/Max`; when
 * that is null the caller wants the base autoscale back (restore), so we clear
 * the price target on the last frame.
 */
export function animateCamera(args: AnimateCameraArgs): () => void {
  const {
    chart,
    from,
    to,
    setPriceTarget,
    durationMs = CAMERA_TWEEN_MS,
    reducedMotion = false,
    onDone,
    now = () => performance.now(),
    raf = (cb) => requestAnimationFrame(cb),
  } = args;

  const toPrice =
    to.priceMin != null && to.priceMax != null
      ? { min: to.priceMin, max: to.priceMax }
      : null;

  const applyFinal = () => {
    chart.setVisibleRange({ from: to.from, to: to.to });
    setPriceTarget(toPrice);
    onDone?.();
  };

  if (reducedMotion || durationMs <= 0) {
    applyFinal();
    return () => {};
  }

  // Interpolate Y only when both ends have a price window; otherwise (a restore to
  // base autoscale) we still tween X and hand Y back to autoscale at the end.
  const canTweenPrice =
    from.priceMin != null &&
    from.priceMax != null &&
    to.priceMin != null &&
    to.priceMax != null;

  const start = now();
  let cancelled = false;

  const tick = () => {
    if (cancelled) return;
    const elapsed = now() - start;
    const raw = clamp(elapsed / durationMs, 0, 1);
    const e = easeInOutCubic(raw);
    chart.setVisibleRange({
      from: lerp(from.from, to.from, e),
      to: lerp(from.to, to.to, e),
    });
    if (canTweenPrice) {
      setPriceTarget({
        min: lerp(from.priceMin!, to.priceMin!, e),
        max: lerp(from.priceMax!, to.priceMax!, e),
      });
    }
    if (raw >= 1) {
      applyFinal();
      return;
    }
    raf(tick);
  };
  raf(tick);

  return () => {
    cancelled = true;
  };
}

/** Convenience: seconds → lightweight-charts UTCTimestamp cast. */
export function asTime(sec: number): UTCTimestamp {
  return sec as UTCTimestamp;
}
export type { Time };
