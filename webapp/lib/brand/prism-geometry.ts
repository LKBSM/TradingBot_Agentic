/**
 * The prism geometry — the ONE place the logo's coordinates live.
 *
 * The React component (`MiaLogo`) and the build-time image generators (favicon,
 * apple-icon, Open Graph card, hosted email PNG) all import from here, so a
 * coordinate is written exactly once. These values are verbatim from the brand
 * source files in `public/brand/*.svg` — do not edit them; they define the mark.
 *
 * Static assets that cannot import JS (`public/icon.svg`, `public/brand/*.svg`)
 * necessarily repeat the path; those are art files, not code.
 */

/** Full three-beam prism, native 120×100 viewBox. */
export const PRISM_RECT = { x: 0, y: 46, width: 29, height: 12 } as const;
export const PRISM_TRIANGLE = 'M46,14 L78,82 L14,82 Z';
export const PRISM_BEAMS = [
  { points: '60,44 62.5,49 118,26 118,12', opacity: 0.55 },
  { points: '63,50 65,55 118,58 118,44', opacity: 1 },
  { points: '66,56 68,61 118,90 118,76', opacity: 0.55 },
] as const;

/** Compact single-beam prism, native 100×100 viewBox (favicon / avatar). */
export const COMPACT_RECT = { x: 0, y: 46, width: 26, height: 12 } as const;
export const COMPACT_TRIANGLE = 'M48,16 L78,84 L18,84 Z';
export const COMPACT_BEAM = '62,48 64,54 100,60 100,42';
