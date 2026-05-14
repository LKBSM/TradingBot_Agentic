/** @type {import('next').NextConfig} */
const createNextIntlPlugin = require('next-intl/plugin');

const withNextIntl = createNextIntlPlugin('./i18n.ts');

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // We render under a separate Vercel deployment; FastAPI backend stays on
  // Railway at api.smartsentinel.ai. Rewrites send /api/* through to it.
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
        source: '/:path*',
        headers: [
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
          {
            key: 'Permissions-Policy',
            value: 'geolocation=(), camera=(), microphone=()',
          },
        ],
      },
    ];
  },
};

module.exports = withNextIntl(nextConfig);
