/**
 * MiaLogo — the SINGLE source of truth for the M.I.A Markets logo.
 *
 * The mark is a prism: a beam enters from the left, crosses a solid triangle,
 * and exits as three stepped beams on the right. Every coordinate and colour
 * below is verbatim from the brand source files in `public/brand/*.svg`
 * (mia-marque / -fond-sombre / -mono, mia-favicon, mia-verrouillage-*). Do not
 * redraw or recolour — surfaces that need a new shape add a variant here, they
 * never copy the path.
 *
 * Tones
 *  · "auto" (default) — picks the brand blue per active theme via CSS vars
 *    (--brand-mark / --brand-word, defined in globals.css). SSR-safe, no JS,
 *    no layout shift: light themes render #2962FF, dark themes #7DA3FF.
 *  · "color" / "dark" / "mono" — force a fixed tone (for a known background:
 *    the dark social card, the monochrome footer, the fixed-colour favicon).
 *
 * Accessibility: `role="img"` with the "M.I.A Markets" label by default. When
 * the logo sits next to text that already says the name, pass `decorative` so
 * it is hidden from screen readers instead of read twice.
 */
import * as React from 'react';
import {
  PRISM_RECT,
  PRISM_TRIANGLE,
  PRISM_BEAMS,
  COMPACT_RECT,
  COMPACT_TRIANGLE,
  COMPACT_BEAM,
} from '@/lib/brand/prism-geometry';

type Tone = 'auto' | 'color' | 'dark' | 'mono';
type Variant = 'mark' | 'horizontal' | 'stacked' | 'compact';

const FIXED_FILL: Record<Exclude<Tone, 'auto'>, string> = {
  color: '#2962FF',
  dark: '#7DA3FF',
  mono: 'currentColor',
};

function markFill(tone: Tone): string {
  return tone === 'auto' ? 'var(--brand-mark, #2962FF)' : FIXED_FILL[tone];
}

function wordFill(tone: Tone): string {
  if (tone === 'auto') return 'var(--brand-word, #0F1729)';
  if (tone === 'dark') return '#FFFFFF';
  if (tone === 'mono') return 'currentColor';
  return '#0F1729';
}

const FONT =
  "Inter, Manrope, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

/** The prism at its native 120×100 coordinate space. */
function Prism({ fill }: { fill: string }) {
  return (
    <g>
      <rect {...PRISM_RECT} fill={fill} />
      <path d={PRISM_TRIANGLE} fill={fill} />
      {PRISM_BEAMS.map((b) => (
        <polygon key={b.points} points={b.points} fill={fill} opacity={b.opacity} />
      ))}
    </g>
  );
}

export interface MiaLogoProps {
  variant?: Variant;
  tone?: Tone;
  /** Rendered height in px; width follows the fixed aspect ratio. */
  height?: number;
  title?: string;
  className?: string;
  /** Hide from assistive tech (logo repeats adjacent visible text). */
  decorative?: boolean;
}

export function MiaLogo({
  variant = 'mark',
  tone = 'auto',
  height = 32,
  title = 'M.I.A Markets',
  className,
  decorative = false,
}: MiaLogoProps) {
  const fill = markFill(tone);
  const textFill = wordFill(tone);

  // A11y: either a labelled image, or hidden decoration next to real text.
  const a11y = decorative
    ? ({ 'aria-hidden': true } as const)
    : ({ role: 'img', 'aria-label': title } as const);

  if (variant === 'compact') {
    return (
      <svg viewBox="0 0 100 100" height={height} className={className} {...a11y}>
        {!decorative && <title>{title}</title>}
        <rect {...COMPACT_RECT} fill={fill} />
        <path d={COMPACT_TRIANGLE} fill={fill} />
        <polygon points={COMPACT_BEAM} fill={fill} />
      </svg>
    );
  }

  if (variant === 'horizontal') {
    return (
      <svg viewBox="0 0 400 100" height={height} className={className} {...a11y}>
        {!decorative && <title>{title}</title>}
        <g transform="translate(8,20) scale(0.6)">
          <Prism fill={fill} />
        </g>
        <text
          x="102"
          y="62"
          fontFamily={FONT}
          fontSize="34"
          fontWeight="500"
          letterSpacing="-0.8"
          fill={textFill}
        >
          M.I.A Markets
        </text>
      </svg>
    );
  }

  if (variant === 'stacked') {
    return (
      <svg viewBox="0 0 260 190" height={height} className={className} {...a11y}>
        {!decorative && <title>{title}</title>}
        <g transform="translate(70,10)">
          <Prism fill={fill} />
        </g>
        <text
          x="130"
          y="160"
          textAnchor="middle"
          fontFamily={FONT}
          fontSize="34"
          fontWeight="500"
          letterSpacing="-0.8"
          fill={textFill}
        >
          M.I.A Markets
        </text>
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 120 100" height={height} className={className} {...a11y}>
      {!decorative && <title>{title}</title>}
      <Prism fill={fill} />
    </svg>
  );
}

export default MiaLogo;
