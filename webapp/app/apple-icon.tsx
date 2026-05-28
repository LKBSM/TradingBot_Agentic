import { ImageResponse } from 'next/og';

// Apple touch icon — used when the user adds the site to their iOS home
// screen. 180×180 PNG, rounded corners are applied by iOS itself so we
// only draw the gold square with the "M" mark centred.
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
          background: 'linear-gradient(135deg, #FBBF24 0%, #B45309 100%)',
          color: '#ffffff',
          fontSize: 120,
          fontWeight: 800,
          fontFamily: 'system-ui',
        }}
      >
        M
      </div>
    ),
    { ...size },
  );
}
