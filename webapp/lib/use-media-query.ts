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

/**
 * Stacked-layout breakpoint — below Tailwind's `xl` (1280px). Phone AND tablet
 * (iPad portrait 834 / landscape 1024) use the single-column tabbed workspace;
 * the three-column desktop grid is reserved for `xl+`, where the two fixed rails
 * (240 + 360px) leave the centre reading column a usable width. Below 1280 the
 * three columns would crush the centre (~138px @834, ~312px @1024).
 */
export const STACKED_QUERY = '(max-width: 1279px)';

export function useStackedLayout(): boolean {
  return useMediaQuery(STACKED_QUERY);
}
