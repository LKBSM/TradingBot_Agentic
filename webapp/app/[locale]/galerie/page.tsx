import { notFound } from 'next/navigation';
import type { Metadata } from 'next';
import { DesignGallery } from '@/components/gallery/DesignGallery';

/**
 * DS-1 — dev-only design gallery route.
 *
 * Renders every Palier-0 presentation component in all its states from the
 * frozen real-data samples (no network, no backend). It is the surface captured
 * for Claude Design and a fast local preview of the components.
 *
 * NOT reachable in production: `notFound()` fires whenever NODE_ENV is
 * production, so a deployed build 404s this path. Kept out of the sitemap and
 * flagged noindex for good measure.
 */
export const metadata: Metadata = {
  title: 'Galerie de composants (dev)',
  robots: { index: false, follow: false },
};

export default function GalleryPage() {
  if (process.env.NODE_ENV === 'production') {
    notFound();
  }
  return <DesignGallery />;
}
