/**
 * BrandMark — the MIA Markets logo mark: three candlesticks (short · tall ·
 * medium) reading left-to-right, echoing a price structure. Legible at favicon
 * size, monochrome-friendly, and reused across the nav and footer. The baseline
 * ("Multi-asset Intelligence Assistant") is NEVER stacked in the bar — it rides
 * a tooltip here and appears in full in the footer.
 */
export function BrandMark({ size = 28 }: { size?: number }) {
  return (
    <span
      aria-hidden
      className="inline-flex items-center justify-center rounded-md bg-gradient-to-br from-amber-400 to-amber-600 shadow-sm"
      style={{ width: size, height: size }}
    >
      <svg
        viewBox="0 0 24 24"
        width={size * 0.72}
        height={size * 0.72}
        fill="none"
        role="img"
      >
        <rect x="3.5" y="9" width="3" height="9" rx="1.4" fill="#04101f" />
        <rect x="10.5" y="4" width="3" height="16" rx="1.4" fill="#04101f" />
        <rect x="17.5" y="11" width="3" height="7" rx="1.4" fill="#04101f" />
      </svg>
    </span>
  );
}
