import { ImageResponse } from 'next/og';
import { CANDLES } from '@/lib/brand/candle-geometry';

/**
 * Stable hosted PNG of the horizontal M.I.A Markets lockup, for transactional
 * emails (verification, password reset, renewal notice). Email clients block
 * SVG, so the backend references this URL — {APP_PUBLIC_URL}/brand/email-logo.png
 * — as an <img> with an alt text. Rendered on a white tile so it reads on the
 * light background every mail client uses, and cached hard since the artwork is
 * immutable per deploy.
 */
export const runtime = 'nodejs';

const SIZE = { width: 480, height: 120 };

export function GET() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          gap: 20,
          padding: '0 32px',
          background: '#ffffff',
        }}
      >
        {/* Brass candle mark + near-black wordmark on the white email tile. */}
        <svg width="90" height="72" viewBox="0 0 90 72" fill="none">
          <g fill="#C9A14A" stroke="#C9A14A">
            {CANDLES.map((c) => (
              <g key={c.wick.x1}>
                <line x1={c.wick.x1} y1={c.wick.y1} x2={c.wick.x2} y2={c.wick.y2} strokeWidth={c.wick.width} opacity={c.opacity} />
                <rect x={c.body.x} y={c.body.y} width={c.body.width} height={c.body.height} rx={1} stroke="none" opacity={c.opacity} />
              </g>
            ))}
          </g>
        </svg>
        <span style={{ fontSize: 38, fontWeight: 500, letterSpacing: 3, color: '#1A1917' }}>
          M.I.A MARKETS
        </span>
      </div>
    ),
    {
      ...SIZE,
      headers: { 'Cache-Control': 'public, max-age=31536000, immutable' },
    },
  );
}
