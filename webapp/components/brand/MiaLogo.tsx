/**
 * MiaLogo — the SINGLE source of truth for the M.I.A Markets logo.
 *
 * The mark is five candlesticks, tallest at the centre and stepping down on both
 * sides. Every coordinate lives once, in `lib/brand/candle-geometry.ts`; the
 * build-time image generators import the same data. Do not redraw or recolour —
 * surfaces that need a new shape add a variant here, they never copy the path.
 *
 * TWO COLOURS, on purpose:
 *  · the candles take `var(--brand-mark)` — brass on all four themes, light
 *    theme included;
 *  · the wordmark takes `var(--brand-word)` — white on the three dark themes,
 *    near-black on the light Parchemin theme.
 * Never merge the two tokens, and never render the wordmark in brass.
 *
 * SSR-safe, no JS, no layout shift: the colour is a CSS variable resolved by the
 * active theme; the aspect ratio is fixed by the viewBox.
 *
 * Accessibility: `role="img"` with the "M.I.A Markets" label by default. When the
 * logo sits next to text that already says the name, pass `decorative` so it is
 * hidden from screen readers instead of read twice.
 */
import * as React from 'react';
import { CANDLES, COMPACT_CANDLES, type Candle } from '@/lib/brand/candle-geometry';

type Variant = 'mark' | 'horizontal' | 'stacked' | 'compact';

const FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

/** Draws a candle set in brass (`--brand-mark`). Symmetry lives in the data. */
function Candles({ candles }: { candles: readonly Candle[] }) {
  return (
    <g fill="var(--brand-mark)" stroke="var(--brand-mark)">
      {candles.map((c) => (
        <React.Fragment key={c.wick.x1}>
          <line
            x1={c.wick.x1}
            y1={c.wick.y1}
            x2={c.wick.x2}
            y2={c.wick.y2}
            strokeWidth={c.wick.width}
            opacity={c.opacity}
          />
          <rect
            x={c.body.x}
            y={c.body.y}
            width={c.body.width}
            height={c.body.height}
            rx={1}
            stroke="none"
            opacity={c.opacity}
          />
        </React.Fragment>
      ))}
    </g>
  );
}

export interface MiaLogoProps {
  variant?: Variant;
  /** Rendered height in px; width follows the fixed aspect ratio. */
  height?: number;
  title?: string;
  className?: string;
  /** Hide from assistive tech (logo repeats adjacent visible text). */
  decorative?: boolean;
}

export function MiaLogo({
  variant = 'mark',
  height = 32,
  title = 'M.I.A Markets',
  className,
  decorative = false,
}: MiaLogoProps) {
  // A11y: either a labelled image, or hidden decoration next to real text.
  const a11y = decorative
    ? ({ 'aria-hidden': true } as const)
    : ({ role: 'img', 'aria-label': title } as const);

  if (variant === 'compact') {
    return (
      <svg viewBox="0 0 72 72" height={height} className={className} {...a11y}>
        {!decorative && <title>{title}</title>}
        <Candles candles={COMPACT_CANDLES} />
      </svg>
    );
  }

  if (variant === 'horizontal') {
    return (
      <svg viewBox="0 0 420 72" height={height} className={className} {...a11y}>
        {!decorative && <title>{title}</title>}
        <g transform="translate(0,8) scale(0.78)">
          <Candles candles={CANDLES} />
        </g>
        <text
          x="94"
          y="45"
          fontFamily={FONT}
          fontSize="27"
          fontWeight="500"
          letterSpacing="3.5"
          fill="var(--brand-word)"
        >
          M.I.A MARKETS
        </text>
      </svg>
    );
  }

  if (variant === 'stacked') {
    return (
      <svg viewBox="0 0 300 150" height={height} className={className} {...a11y}>
        {!decorative && <title>{title}</title>}
        <g transform="translate(105,14)">
          <Candles candles={CANDLES} />
        </g>
        <text
          x="150"
          y="125"
          textAnchor="middle"
          fontFamily={FONT}
          fontSize="25"
          fontWeight="500"
          letterSpacing="3.2"
          fill="var(--brand-word)"
        >
          M.I.A MARKETS
        </text>
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 90 72" height={height} className={className} {...a11y}>
      {!decorative && <title>{title}</title>}
      <Candles candles={CANDLES} />
    </svg>
  );
}

export default MiaLogo;
