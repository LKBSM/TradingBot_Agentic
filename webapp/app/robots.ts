import type { MetadataRoute } from 'next';

const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ?? 'https://mia.markets';

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: '*',
        // Allow indexing of the public landing. Block inactive locales (302
        // redirect → /), API routes, Next internals, and the dynamic OG image
        // (only crawled via og:image meta — direct indexing has no value).
        allow: '/',
        disallow: [
          '/api/',
          '/en/',
          '/de/',
          '/es/',
          '/_next/',
          '/opengraph-image',
        ],
      },
    ],
    sitemap: `${SITE_URL}/sitemap.xml`,
    host: SITE_URL,
  };
}
