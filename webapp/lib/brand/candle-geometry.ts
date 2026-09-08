/**
 * The M.I.A Markets mark — five candlesticks — the ONE place the coordinates live.
 *
 * The React component (`MiaLogo`) and the build-time image generators (favicon,
 * apple-icon, Open Graph card, hosted email PNG) all import from here, so a
 * coordinate is written exactly once. These values are verbatim from the brand
 * source files in `public/brand/*.svg` — do not edit them; they define the mark.
 *
 * SYMMETRY IS STRUCTURAL, NOT DECORATIVE. The five heights mirror around the
 * centre candle (the tallest): two medium, two short. They never increase or
 * decrease monotonically from left to right — an ascending run would read as a
 * direction, i.e. a prediction, which the product forbids. This symmetry is what
 * makes the mark usable. A test guards it; do not "tidy" it away.
 *
 * TWO COLOURS, on purpose: candles take `--brand-mark` (brass on all four
 * themes); the wordmark takes `--brand-word` (white on the dark themes,
 * near-black on the light Parchemin theme). Never merge the two tokens.
 *
 * Static assets that cannot import JS (`public/icon.svg`, `public/brand/*.svg`)
 * necessarily repeat these coordinates; those are art files, not code.
 */

export interface Candle {
  wick: { x1: number; y1: number; x2: number; y2: number; width: number };
  body: { x: number; y: number; width: number; height: number };
  opacity: number;
}

/** Full five-candle mark, native 90×72 viewBox. Heights mirror the centre. */
export const CANDLES: readonly Candle[] = [
  { wick: { x1: 7, y1: 22, x2: 7, y2: 50, width: 2.5 }, body: { x: 1.5, y: 27, width: 11, height: 18 }, opacity: 0.45 },
  { wick: { x1: 26, y1: 12, x2: 26, y2: 60, width: 2.5 }, body: { x: 20.5, y: 20, width: 11, height: 32 }, opacity: 0.7 },
  { wick: { x1: 45, y1: 5, x2: 45, y2: 67, width: 3 }, body: { x: 39, y: 15, width: 12, height: 42 }, opacity: 1 },
  { wick: { x1: 64, y1: 12, x2: 64, y2: 60, width: 2.5 }, body: { x: 58.5, y: 20, width: 11, height: 32 }, opacity: 0.7 },
  { wick: { x1: 83, y1: 22, x2: 83, y2: 50, width: 2.5 }, body: { x: 77.5, y: 27, width: 11, height: 18 }, opacity: 0.45 },
] as const;

/**
 * Compact three-candle mark, native 72×72 viewBox — full opacity, no variations.
 * Reserved for small sizes (favicon, app icons, the M.I.A avatar): below 40px the
 * five candles and their opacity steps turn to mush. Still symmetric.
 */
export const COMPACT_CANDLES: readonly Candle[] = [
  { wick: { x1: 13, y1: 22, x2: 13, y2: 50, width: 4 }, body: { x: 6, y: 28, width: 14, height: 16 }, opacity: 1 },
  { wick: { x1: 36, y1: 6, x2: 36, y2: 66, width: 5 }, body: { x: 28, y: 16, width: 16, height: 40 }, opacity: 1 },
  { wick: { x1: 59, y1: 22, x2: 59, y2: 50, width: 4 }, body: { x: 52, y: 28, width: 14, height: 16 }, opacity: 1 },
] as const;
