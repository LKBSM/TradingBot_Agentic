import { ImageResponse } from 'next/og';
import { COMPACT_CANDLES } from '@/lib/brand/candle-geometry';

// Apple touch icon — used when the site is added to the iOS home screen.
// 180×180 PNG; iOS applies its own rounded mask, so we fill the tile and
// centre the compact three-candle mark (mia-favicon.svg). Server-generated:
// the brass mark colour (#C9A14A) is a literal here on purpose.
export const size = { width: 180, height: 180 };
export const contentType = 'image/png';

export default function AppleIcon() {
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
        }}
      >
        <svg width="120" height="120" viewBox="0 0 72 72" fill="none">
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
