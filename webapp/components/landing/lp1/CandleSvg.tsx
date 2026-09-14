import { buildCandles, priceBounds, xForIndex, yForPrice, type Bounds } from './chart';
import { DEMO_CLOSES } from './data';
import styles from './lp1.module.css';

interface Props {
  width: number;
  height: number;
  closes?: readonly number[];
  extra?: readonly number[];
  /** decorative only */
  ariaHidden?: boolean;
  /**
   * LP-3 hero: draw the series left to right instead of all at once. Purely a
   * CSS reveal — every candle is in the DOM from the first (server) paint, it
   * is only its opacity that is delayed, and only when the visitor has not asked
   * for reduced motion. So a visitor with no JS, or with reduced motion, sees
   * the finished chart.
   */
  stagger?: boolean;
}

/** Deterministic candlestick SVG for the illustration charts. Bull/bear colours
 * come from the theme tokens so it matches the product. */
export function CandleSvg({ width, height, closes = DEMO_CLOSES, extra = [], stagger = false }: Props) {
  const candles = buildCandles(closes);
  const bounds: Bounds = priceBounds(candles, extra);
  const n = candles.length;
  const bw = Math.max(2.5, ((width - 76) / n) * 0.6);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true" focusable="false">
      {candles.map((k, i) => {
        const x = xForIndex(i, n, width);
        const color = k.up ? 'var(--bull)' : 'var(--bear)';
        const yH = yForPrice(k.h, bounds, height);
        const yL = yForPrice(k.l, bounds, height);
        const yO = yForPrice(k.o, bounds, height);
        const yC = yForPrice(k.c, bounds, height);
        const top = Math.min(yO, yC);
        const bh = Math.max(1.2, Math.abs(yC - yO));
        return (
          <g
            key={i}
            {...(stagger
              ? { className: styles.cdl, style: { animationDelay: `${i * 22}ms` } }
              : null)}
          >
            <line x1={x} y1={yH} x2={x} y2={yL} stroke={color} strokeWidth="1" />
            <rect x={x - bw / 2} y={top} width={bw} height={bh} fill={color} rx="1" />
          </g>
        );
      })}
    </svg>
  );
}

export { priceBounds, buildCandles };
