/**
 * RG-1d — the vertical price range to pin when a calendar reference level is
 * traced on the chart. Pure + unit-testable (no charting dependency): the chart
 * component sets it imperatively via IPriceScaleApi.setVisibleRange, which is
 * reliable where toggling autoScale is not (the library coalesces an unchanged
 * autoScale value, so the view never moves — the « il ne ramène pas » bug).
 */

// Margin on each side of the union so neither the level nor the current price
// sits on the very edge, and the structure between them stays readable — that
// context is what makes the trace useful (never a tight zoom).
export const REFERENCE_VIEW_MARGIN_FRAC = 0.08;

/**
 * The price range holding the traced `level` AND the extremes of the visible
 * candles (indices `from`..`to`, inclusive), padded by
 * {@link REFERENCE_VIEW_MARGIN_FRAC}. Including the visible candles keeps the
 * current price (always a candle) and the structure between it and the level on
 * screen. Returns `{ from, to }` prices (`from` < `to`) for setVisibleRange, or a
 * small pad around the level when no valid candle sits in range.
 */
export function computeReferenceViewRange(
  level: number,
  candles: readonly { high: number; low: number }[],
  from: number,
  to: number,
): { from: number; to: number } | null {
  if (!Number.isFinite(level)) return null;
  let lo = level;
  let hi = level;
  const lastIdx = candles.length - 1;
  const start = Math.max(0, Math.min(from, lastIdx));
  const end = Math.max(0, Math.min(to, lastIdx));
  for (let i = start; i <= end; i++) {
    const c = candles[i];
    if (!c) continue;
    if (Number.isFinite(c.low)) lo = Math.min(lo, c.low);
    if (Number.isFinite(c.high)) hi = Math.max(hi, c.high);
  }
  if (!(hi > lo)) {
    // Degenerate (level on a flat span / no candles): pad around the level.
    const pad = Math.abs(level) * 0.001 || 1;
    return { from: level - pad, to: level + pad };
  }
  const pad = (hi - lo) * REFERENCE_VIEW_MARGIN_FRAC;
  return { from: lo - pad, to: hi + pad };
}
