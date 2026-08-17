import { ImageResponse } from 'next/og';
import { COMPACT_RECT, COMPACT_TRIANGLE, COMPACT_BEAM } from '@/lib/brand/prism-geometry';

// Apple touch icon — used when the site is added to the iOS home screen.
// 180×180 PNG; iOS applies its own rounded mask, so we fill the tile and
// centre the compact single-beam prism (mia-favicon.svg).
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
        <svg width="120" height="120" viewBox="0 0 100 100" fill="none">
          <rect {...COMPACT_RECT} fill="#7DA3FF" />
          <path d={COMPACT_TRIANGLE} fill="#7DA3FF" />
          <polygon points={COMPACT_BEAM} fill="#7DA3FF" />
        </svg>
      </div>
    ),
    { ...size },
  );
}
