import { ImageResponse } from 'next/og';
import { CANDLES } from '@/lib/brand/candle-geometry';

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
          {/* Brass candle mark (#C9A14A on the dark card) + white wordmark. */}
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
