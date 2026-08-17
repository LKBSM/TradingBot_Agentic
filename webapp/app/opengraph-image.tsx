import { ImageResponse } from 'next/og';
import { PRISM_RECT, PRISM_TRIANGLE, PRISM_BEAMS } from '@/lib/brand/prism-geometry';

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
export const alt = 'M.I.A Markets — Indicateur de marché conversationnel';
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
        <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
          {/* Prism mark (dark-background tone #7DA3FF) + wordmark. */}
          <svg width="72" height="60" viewBox="0 0 120 100" fill="none">
            <rect {...PRISM_RECT} fill="#7DA3FF" />
            <path d={PRISM_TRIANGLE} fill="#7DA3FF" />
            {PRISM_BEAMS.map((b) => (
              <polygon key={b.points} points={b.points} fill="#7DA3FF" opacity={b.opacity} />
            ))}
          </svg>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: 30, fontWeight: 600, letterSpacing: -0.5 }}>
              M.I.A Markets
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
            Lectures algorithmiques · chatbot M.I.A Agent · posture éducative
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
