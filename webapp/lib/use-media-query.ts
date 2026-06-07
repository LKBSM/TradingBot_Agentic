'use client';

import * as React from 'react';

/**
 * Subscribe to a CSS media query. SSR-safe: returns `false` during server
 * render and the first client paint, then syncs to the real value after mount
 * (so there is no `window`/`matchMedia` access during SSR). Tests can stub
 * `window.matchMedia` to drive the result.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = React.useState(false);

  React.useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mql = window.matchMedia(query);
    const onChange = () => setMatches(mql.matches);
    onChange();
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, [query]);

  return matches;
}

/** Mobile breakpoint — below Tailwind's `md` (768px). */
export const MOBILE_QUERY = '(max-width: 767px)';

export function useIsMobile(): boolean {
  return useMediaQuery(MOBILE_QUERY);
}
