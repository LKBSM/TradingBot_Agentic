import type { MetadataRoute } from 'next';

/**
 * PWA manifest for M.I.A. Markets. Modern browsers (Chrome 109+, Edge,
 * recent Firefox) accept SVG icons with sizes="any" and purpose="any
 * maskable" — keeps the icon source single-file. Older Android Chrome
 * needs PNG fallbacks (icon.tsx + apple-icon.tsx generate those at
 * build time via Next.js ImageResponse).
 */
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: 'M.I.A. Markets — Indicateur de marché',
    short_name: 'M.I.A.',
    description:
      "M.I.A. Markets · Multi-asset Intelligence Assistant — indicateur de marché conversationnel pour XAU/USD et FX. Posture éducative.",
    start_url: '/',
    scope: '/',
    display: 'standalone',
    orientation: 'portrait-primary',
    background_color: '#0a0f1c',
    theme_color: '#0a0f1c',
    lang: 'fr',
    dir: 'ltr',
    categories: ['finance', 'productivity', 'education'],
    icons: [
      {
        src: '/icon.svg',
        sizes: 'any',
        type: 'image/svg+xml',
        purpose: 'any',
      },
      {
        src: '/icon.svg',
        sizes: 'any',
        type: 'image/svg+xml',
        purpose: 'maskable',
      },
    ],
  };
}
