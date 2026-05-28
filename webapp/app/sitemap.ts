import type { MetadataRoute } from 'next';

// Default canonical base URL. Override via NEXT_PUBLIC_SITE_URL once the
// domain is live so the sitemap and robots emit absolute production URLs.
const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ?? 'https://mia.markets';

const LAST_MODIFIED = new Date('2026-05-27');

export default function sitemap(): MetadataRoute.Sitemap {
  // V1 = FR only. EN/DE/ES routes are inactive (302 → FR via middleware) so
  // they MUST stay out of the sitemap to avoid SEO duplication / soft-404.
  // When EN is activated (V3), add an `alternates.languages` block.
  return [
    {
      url: `${SITE_URL}/`,
      lastModified: LAST_MODIFIED,
      changeFrequency: 'weekly',
      priority: 1,
    },
  ];
}
