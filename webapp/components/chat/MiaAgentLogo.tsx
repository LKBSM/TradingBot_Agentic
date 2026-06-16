import * as React from 'react';

/**
 * M.I.A Agent avatar — a small, sober inline SVG of three candlesticks
 * (Direction 1 language): two bullish (#2F9E78) framing one bearish (#C2693E),
 * thin bodies, discreet wicks. Purely presentational — no logic, no theming
 * tokens — so it reads identically on the light and dark `bg-primary` avatar
 * fill. Replaces the generic lucide `Bot` glyph in the chat header / avatar.
 */
const BULL = '#2F9E78';
const BEAR = '#C2693E';

export function MiaAgentLogo({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={className}
      fill="none"
      role="img"
      aria-hidden
    >
      {/* Left candle — bullish */}
      <line x1="6" y1="4" x2="6" y2="19" stroke={BULL} strokeWidth="1.2" />
      <rect x="4.4" y="8" width="3.2" height="7" rx="0.6" fill={BULL} />
      {/* Center candle — bearish (taller) */}
      <line x1="12" y1="3" x2="12" y2="20.5" stroke={BEAR} strokeWidth="1.2" />
      <rect x="10.4" y="6" width="3.2" height="9.5" rx="0.6" fill={BEAR} />
      {/* Right candle — bullish */}
      <line x1="18" y1="6" x2="18" y2="18" stroke={BULL} strokeWidth="1.2" />
      <rect x="16.4" y="9" width="3.2" height="6" rx="0.6" fill={BULL} />
    </svg>
  );
}
