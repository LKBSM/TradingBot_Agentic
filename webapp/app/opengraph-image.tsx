import { ImageResponse } from 'next/og';

/**
 * Open Graph card — 1200×630 PNG generated at build time via next/og.
 * Used as og:image + twitter:image for Twitter, LinkedIn, WhatsApp,
 * Telegram previews. The composition stays sober (gold accent + verdict-
 * style typography on dark) to match the in-app aesthetic.
 *
 * Edit the copy below carefully — every share reaches a cold prospect, so
 * the wording must be compliance-safe (no "signal", no "gain", no promise).
 */
export const runtime = 'nodejs';
export const alt = 'MIA Markets — Indicateur de marché conversationnel';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: 72,
          background: 'linear-gradient(135deg, #0a0f1c 0%, #111827 100%)',
          color: '#f9fafb',
          fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div
            style={{
              width: 56,
              height: 56,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: 12,
              background: 'linear-gradient(135deg, #FBBF24 0%, #B45309 100%)',
              fontWeight: 800,
              fontSize: 32,
              color: '#fff',
            }}
          >
            M
          </div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: 22, fontWeight: 600, letterSpacing: -0.5 }}>
              MIA Markets
            </span>
            <span
              style={{
                fontSize: 14,
                color: '#9ca3af',
                textTransform: 'uppercase',
                letterSpacing: 1.2,
              }}
            >
              Multi-asset Intelligence Assistant
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <h1
            style={{
              fontSize: 64,
              fontWeight: 600,
              lineHeight: 1.05,
              letterSpacing: -2,
              margin: 0,
              maxWidth: 940,
            }}
          >
            Comprenez le marché — sans qu&apos;on vous dise quoi faire.
          </h1>
          <p
            style={{
              fontSize: 22,
              color: '#9ca3af',
              maxWidth: 900,
              margin: 0,
              lineHeight: 1.4,
            }}
          >
            Lectures algorithmiques · chatbot Sentinel · posture éducative ·
            conforme UE 2024/2811
          </p>
        </div>

        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            paddingTop: 24,
            borderTop: '1px solid #1f2937',
            fontSize: 16,
            color: '#9ca3af',
          }}
        >
          <span>mia.markets</span>
          <span style={{ fontStyle: 'italic' }}>
            Early Access · Educational Use
          </span>
        </div>
      </div>
    ),
    { ...size },
  );
}
