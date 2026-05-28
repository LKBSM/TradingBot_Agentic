/** @type {import('next').NextConfig} */
const createNextIntlPlugin = require('next-intl/plugin');

const withNextIntl = createNextIntlPlugin('./i18n.ts');

const isProd = process.env.NODE_ENV === 'production';

/**
 * Content-Security-Policy — strict, but allows the bits Next.js + Anthropic
 * SDK need to work. `'unsafe-eval'` is enabled in dev only (HMR pipeline).
 * `'unsafe-inline'` for script and style is unavoidable today because
 * Next.js inlines critical hydration data and Tailwind generates inline
 * style attributes on some components. The proper hardening (per-request
 * nonce via middleware) is filed in `docs/frontend/TODO_NEXT_SPRINTS.md`
 * under "harden CSP V3".
 */
const cspDirectives = [
  "default-src 'self'",
  isProd
    ? "script-src 'self' 'unsafe-inline'"
    : "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "connect-src 'self' https://api.anthropic.com",
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

// DG-033 — Sentry wrapper. ``withSentryConfig`` uploads source maps to
// Sentry at build-time when ``SENTRY_AUTH_TOKEN`` is set (CI runner). At
// dev / preview it's a no-op so the local loop stays fast.
const baseConfig = withNextIntl(nextConfig);

const sentryEnabled =
  process.env.SENTRY_DSN && process.env.SENTRY_AUTH_TOKEN;

if (sentryEnabled) {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { withSentryConfig } = require('@sentry/nextjs');
  module.exports = withSentryConfig(baseConfig, {
    org: process.env.SENTRY_ORG,
    project: process.env.SENTRY_PROJECT,
    authToken: process.env.SENTRY_AUTH_TOKEN,
    silent: !process.env.CI,
    // Source maps only on prod build; never expose to public.
    hideSourceMaps: true,
    disableLogger: true,
  });
} else {
  module.exports = baseConfig;
}
