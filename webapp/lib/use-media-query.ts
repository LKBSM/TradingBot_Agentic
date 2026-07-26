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
 * Stacked-layout breakpoint (mission UI-2b) — the phone-only tabbed workspace.
 * The reference single-column reading centre must show from 768px up, so only
 * true phones (<768px) get the tabbed MobileWorkspace. From 768px the shell
 * shows the reference centre and adapts the side columns instead of crushing it:
 * the docked chat becomes an off-canvas drawer between 768 and 1279px, and only
 * ≥1280px keeps the full three-column grid (see shell.css). The rail is hidden
 * by the shell below 768 (MobileWorkspace carries its own Marchés tab).
 */
export const STACKED_QUERY = '(max-width: 767px)';

export function useStackedLayout(): boolean {
  return useMediaQuery(STACKED_QUERY);
}
