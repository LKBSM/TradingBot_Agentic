/** @type {import('next').NextConfig} */
const createNextIntlPlugin = require('next-intl/plugin');

const withNextIntl = createNextIntlPlugin('./i18n.ts');

const isProd = process.env.NODE_ENV === 'production';

/**
 * Content-Security-Policy — strict. `'unsafe-eval'` is enabled in dev only
 * (HMR pipeline). `'unsafe-inline'` for script and style is unavoidable today
 * because Next.js inlines critical hydration data and Tailwind generates inline
 * style attributes on some components. The proper hardening (per-request
 * nonce via middleware) is filed in `docs/frontend/TODO_NEXT_SPRINTS.md`
 * under "harden CSP V3".
 *
 * Chantier 5.A (T5): `connect-src` is `'self'` only. The webapp no longer talks
 * to api.anthropic.com directly — the Anthropic SDK was decommissioned and the
 * chatbot now goes through the same-origin `/api/*` rewrite to the FastAPI
 * backend (which holds the only Anthropic key). Defense in depth.
 */
const cspDirectives = [
  "default-src 'self'",
  isProd
    ? "script-src 'self' 'unsafe-inline'"
    : "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "connect-src 'self'",
  "frame-ancestors 'none'",
  "form-action 'self'",
  "base-uri 'self'",
  "object-src 'none'",
  "manifest-src 'self'",
  ...(isProd ? ['upgrade-insecure-requests'] : []),
].join('; ');

const securityHeaders = [
  // Mitigate clickjacking — DENY rather than SAMEORIGIN because the app is
  // never embedded by design (no SSO popups, no widgets).
  { key: 'X-Frame-Options', value: 'DENY' },
  // Mitigate MIME sniffing in older browsers.
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  // Don't leak the full referer on cross-origin navigations.
  { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
  // HSTS in prod only — applying in dev would break http://localhost.
  ...(isProd
    ? [
        {
          key: 'Strict-Transport-Security',
          value: 'max-age=63072000; includeSubDomains; preload',
        },
      ]
    : []),
  // Lock down browser features we don't use. Add to the deny-list whenever
  // a sensitive API gets added to the surface.
  {
    key: 'Permissions-Policy',
    value: [
      'accelerometer=()',
      'autoplay=()',
      'camera=()',
      'display-capture=()',
      'encrypted-media=()',
      'fullscreen=(self)',
      'geolocation=()',
      'gyroscope=()',
      'microphone=()',
      'midi=()',
      'payment=()',
      'picture-in-picture=()',
      'publickey-credentials-get=()',
      'screen-wake-lock=()',
      'usb=()',
      'web-share=(self)',
      'xr-spatial-tracking=()',
    ].join(', '),
  },
  // Defence in depth even though CSP is the real backstop.
  { key: 'X-DNS-Prefetch-Control', value: 'on' },
  // Modern recommendation — disable legacy XSS auditor (rely on CSP).
  { key: 'X-XSS-Protection', value: '0' },
  // Cross-origin isolation — protects against Spectre-style leaks if we
  // ever add SharedArrayBuffer or fine-grained timers. Safe defaults.
  { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
  { key: 'Cross-Origin-Resource-Policy', value: 'same-site' },
  { key: 'Content-Security-Policy', value: cspDirectives },
];

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // Emit a self-contained server bundle (.next/standalone) so the Docker image
  // ships only the traced runtime deps + a tiny node server, not the full
  // node_modules. Enables the slim multi-stage webapp/Dockerfile.
  output: 'standalone',
  // Disable Next's built-in gzip. It re-compresses the rewrite-proxied SSE
  // stream (GET /api/live-price), buffering the small per-tick frames until its
  // threshold — so the browser's EventSource opened but received nothing, while
  // curl (no Accept-Encoding by default) was unaffected. An SSE stream must
  // never be gzip-buffered. Normal asset compression is handled at the edge/CDN
  // in production, so turning this off here is safe.
  compress: false,
  // Vercel deployment fronts the FastAPI backend on Fly.io at
  // api.mia.markets. Rewrites send /api/* through to it (set
  // NEXT_PUBLIC_API_BASE for non-default targets).
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination:
          process.env.NEXT_PUBLIC_API_BASE
            ? `${process.env.NEXT_PUBLIC_API_BASE}/api/:path*`
            : 'http://localhost:8000/api/:path*',
      },
    ];
  },
  async headers() {
    return [
      {
        // Apply to everything except Next internals + static assets that
        // should be cacheable downstream (immutable hash chunks).
        source: '/:path*',
        headers: securityHeaders,
      },
    ];
  },
  // Allow remote image patterns for next/image as we add OG/marketing assets.
  // V1 ships zero remote images; the list stays empty.
  images: {
    remotePatterns: [],
  },
};

module.exports = withNextIntl(nextConfig);
