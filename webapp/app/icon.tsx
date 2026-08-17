import { ImageResponse } from 'next/og';
import { COMPACT_RECT, COMPACT_TRIANGLE, COMPACT_BEAM } from '@/lib/brand/prism-geometry';

// Favicon — the compact single-beam prism (mia-favicon.svg) on a dark tile.
// The full three-beam mark turns to mush at 16px, which is exactly why the
// compact variant exists. Next.js auto-wires this as <link rel="icon">.
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
        <svg width="22" height="22" viewBox="0 0 100 100" fill="none">
          <rect {...COMPACT_RECT} fill="#7DA3FF" />
          <path d={COMPACT_TRIANGLE} fill="#7DA3FF" />
          <polygon points={COMPACT_BEAM} fill="#7DA3FF" />
        </svg>
      </div>
    ),
    { ...size },
  );
}
