import { ImageResponse } from 'next/og';
import { COMPACT_CANDLES } from '@/lib/brand/candle-geometry';

// Favicon — the compact three-candle mark (mia-favicon.svg) on a dark tile.
// The full five-candle mark turns to mush at 16px, which is exactly why the
// compact variant exists. Next.js auto-wires this as <link rel="icon">.
// Server-generated image: the brass mark colour (#C9A14A) is a literal here on
// purpose — there is no theme at favicon time.
export const size = { width: 32, height: 32 };
export const contentType = 'image/png';

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#0a0f1c',
          borderRadius: 6,
        }}
      >
        <svg width="24" height="24" viewBox="0 0 72 72" fill="none">
          <g fill="#C9A14A" stroke="#C9A14A">
            {COMPACT_CANDLES.map((c) => (
              <g key={c.wick.x1}>
                <line x1={c.wick.x1} y1={c.wick.y1} x2={c.wick.x2} y2={c.wick.y2} strokeWidth={c.wick.width} />
                <rect x={c.body.x} y={c.body.y} width={c.body.width} height={c.body.height} rx={1} stroke="none" />
              </g>
            ))}
          </g>
        </svg>
      </div>
    ),
    { ...size },
  );
}
