import '@testing-library/jest-dom/vitest';
import { configure } from '@testing-library/dom';

// The jsdom environment can be slow under parallel file execution on some
// machines; give async utilities (waitFor / findBy*) a larger budget so
// timing-sensitive tests don't flake under load.
configure({ asyncUtilTimeout: 5000 });

/**
 * jsdom ships no `window.matchMedia`, and product code legitimately asks for
 * `prefers-reduced-motion` (the landing hero, the chart, the auth backdrop).
 * Without a default every render of those components throws — a gap in the
 * environment, not a behaviour worth asserting. The default answers "no
 * preference" (matches: false); a test that cares stubs its own via
 * `vi.stubGlobal('matchMedia', …)`, which still wins.
 */
if (typeof window !== 'undefined' && !window.matchMedia) {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
}
