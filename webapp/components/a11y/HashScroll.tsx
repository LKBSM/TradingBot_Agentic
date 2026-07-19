'use client';

import { usePathname } from 'next/navigation';
import * as React from 'react';

/**
 * Scrolls to the URL hash target after a client-side navigation (NAV-08).
 *
 * The native hash jump fires before the target section has mounted when you
 * arrive on the landing from another route (e.g. footer "FAQ" → `/#faq`), so the
 * scroll is silently dropped. This retries for a few frames until the element
 * `#<hash>` exists, then smooth-scrolls to it. No-op when there is no hash.
 */
export function HashScroll() {
  const pathname = usePathname();

  React.useEffect(() => {
    if (typeof window === 'undefined') return;
    const hash = window.location.hash;
    if (!hash || hash.length < 2) return;
    const id = decodeURIComponent(hash.slice(1));

    let raf = 0;
    let tries = 0;
    const tick = () => {
      const el = document.getElementById(id);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        return;
      }
      // Give the section a handful of frames to mount after navigation.
      if (tries++ < 12) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [pathname]);

  return null;
}
