import { ImageResponse } from 'next/og';
import { PRISM_RECT, PRISM_TRIANGLE, PRISM_BEAMS } from '@/lib/brand/prism-geometry';

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
        <svg width="86" height="72" viewBox="0 0 120 100" fill="none">
          <rect {...PRISM_RECT} fill="#2962FF" />
          <path d={PRISM_TRIANGLE} fill="#2962FF" />
          {PRISM_BEAMS.map((b) => (
            <polygon key={b.points} points={b.points} fill="#2962FF" opacity={b.opacity} />
          ))}
        </svg>
        <span style={{ fontSize: 40, fontWeight: 600, letterSpacing: -1, color: '#0F1729' }}>
          M.I.A Markets
        </span>
      </div>
    ),
    {
      ...SIZE,
      headers: { 'Cache-Control': 'public, max-age=31536000, immutable' },
    },
  );
}
