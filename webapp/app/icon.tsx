import { ImageResponse } from 'next/og';

// Static favicon — gold-gradient "M" mark. Next.js auto-detects this file
// and wires <link rel="icon"> in the document head.
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
          background: 'linear-gradient(135deg, #FBBF24 0%, #B45309 100%)',
          color: '#ffffff',
          fontSize: 22,
          fontWeight: 800,
          fontFamily: 'system-ui',
          borderRadius: 6,
        }}
      >
        M
      </div>
    ),
    { ...size },
  );
}
