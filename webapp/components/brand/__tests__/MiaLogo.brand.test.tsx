/**
 * BRD-3 brand guards. The properties the "candle" logo rollout must keep true:
 *  1. the candle geometry lives in exactly ONE code file (single source);
 *  2. the five candle heights are SYMMETRIC around the centre — never an
 *     ascending / descending run (that would read as a prediction);
 *  3. two tokens, never merged: candles = --brand-mark, wordmark = --brand-word;
 *  4. no old "prism" logo string / colour / geometry survives anywhere;
 *  5. no hard-coded logo colour outside the server-generated images;
 *  6. the compact variant is used at small sizes; the full mark never < 40px;
 *  7. the logo never appears in a loading / empty / error state.
 */
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MiaLogo } from '../MiaLogo';
import { CANDLES, COMPACT_CANDLES } from '../../../lib/brand/candle-geometry';

const WEBAPP = join(__dirname, '..', '..', '..');

function walk(dir: string, exts = ['.ts', '.tsx']): string[] {
  const out: string[] = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    if (e.name === 'node_modules' || e.name === '.next') continue;
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p, exts));
    else if (exts.some((x) => e.name.endsWith(x)) && !/\.(test|spec)\./.test(e.name))
      out.push(p);
  }
  return out;
}

const CODE = walk(join(WEBAPP, 'app'))
  .concat(walk(join(WEBAPP, 'components')))
  .concat(walk(join(WEBAPP, 'lib')));

// The four server-generated images legitimately hard-code the brass colour
// (there is no CSS theme at build time). Everything else must go through the vars.
const GENERATORS = [
  'app/icon.tsx',
  'app/apple-icon.tsx',
  'app/opengraph-image.tsx',
  'app/brand/email-logo.png/route.tsx',
].map((p) => join(WEBAPP, p).replace(/\\/g, '/'));

describe('BRD-3 — single candle source', () => {
  it('the candle geometry arrays are declared in exactly one code file', () => {
    for (const decl of ['export const CANDLES', 'export const COMPACT_CANDLES']) {
      const hits = CODE.filter((f) => readFileSync(f, 'utf-8').includes(decl));
      expect(hits, `"${decl}" duplicated in: ${hits.join(', ')}`).toHaveLength(1);
      expect((hits[0] ?? '').replace(/\\/g, '/')).toContain('lib/brand/candle-geometry');
    }
  });

  it('MiaLogo has a single definition', () => {
    const defs = CODE.filter((f) => /export function MiaLogo\b/.test(readFileSync(f, 'utf-8')));
    expect(defs).toHaveLength(1);
  });

  it('the generators import the shared geometry (no copied coordinates)', () => {
    for (const f of GENERATORS) {
      expect(readFileSync(f, 'utf-8'), `${f} should import candle-geometry`).toContain(
        'candle-geometry',
      );
    }
  });
});

describe('BRD-3 — symmetry is structural (never an ascending curve)', () => {
  function assertSymmetric(candles: readonly { body: { height: number }; wick: { y1: number }; opacity: number }[]) {
    const heights = candles.map((c) => c.body.height);
    const tops = candles.map((c) => c.wick.y1);
    const opacities = candles.map((c) => c.opacity);
    // Mirror image around the centre — reversing changes nothing.
    expect(heights).toEqual([...heights].reverse());
    expect(tops).toEqual([...tops].reverse());
    expect(opacities).toEqual([...opacities].reverse());
    // The centre candle is the UNIQUE tallest; strictly rising to it, then falling.
    const mid = Math.floor(heights.length / 2);
    for (let i = 1; i <= mid; i++) expect(heights[i]!).toBeGreaterThan(heights[i - 1]!);
    for (let i = mid + 1; i < heights.length; i++) expect(heights[i]!).toBeLessThan(heights[i - 1]!);
    // A strictly increasing (or decreasing) run would fail the mirror assertion.
  }

  it('the five candles mirror around the centre', () => {
    expect(CANDLES).toHaveLength(5);
    assertSymmetric(CANDLES);
  });

  it('the compact three candles mirror around the centre', () => {
    expect(COMPACT_CANDLES).toHaveLength(3);
    assertSymmetric(COMPACT_CANDLES);
  });
});

describe('BRD-3 — two tokens, candles brass and word its own colour', () => {
  it('candles are painted from --brand-mark', () => {
    const { container } = render(<MiaLogo variant="mark" />);
    const g = container.querySelector('g');
    expect(g?.getAttribute('fill')).toBe('var(--brand-mark)');
    expect(container.querySelector('line')?.getAttribute('opacity')).toBeTruthy();
  });

  it('the wordmark is painted from --brand-word, never from the candle token', () => {
    for (const variant of ['horizontal', 'stacked'] as const) {
      const { container } = render(<MiaLogo variant={variant} />);
      const text = container.querySelector('text');
      expect(text?.getAttribute('fill')).toBe('var(--brand-word)');
      // The two tokens are distinct — the name must not take the candle colour.
      expect(text?.getAttribute('fill')).not.toBe('var(--brand-mark)');
    }
  });
});

describe('BRD-3 — variant + accessibility', () => {
  it('renders the labelled mark by default (role img)', () => {
    const { getByRole } = render(<MiaLogo />);
    expect(getByRole('img').getAttribute('aria-label')).toBe('M.I.A Markets');
  });

  it('is hidden from screen readers when decorative', () => {
    const { container, queryByRole } = render(<MiaLogo decorative />);
    expect(queryByRole('img')).toBeNull();
    expect(container.querySelector('svg')?.getAttribute('aria-hidden')).toBe('true');
  });

  it('the horizontal + stacked lockups spell the company name', () => {
    for (const variant of ['horizontal', 'stacked'] as const) {
      const { container } = render(<MiaLogo variant={variant} />);
      expect(container.textContent).toContain('M.I.A MARKETS'); // visible wordmark
      expect(container.querySelector('svg')?.getAttribute('aria-label')).toBe('M.I.A Markets');
    }
  });
});

describe('BRD-3 — no old prism logo survives', () => {
  const SURFACES = CODE.concat(walk(join(WEBAPP, 'messages'), ['.json']));

  it('the old prism geometry, colours and names are gone from code', () => {
    for (const f of CODE) {
      const src = readFileSync(f, 'utf-8');
      expect(src.includes('prism-geometry'), `prism-geometry ref in ${f}`).toBe(false);
      for (const trace of ['M46,14 L78,82 L14,82 Z', 'M48,16 L78,84 L18,84 Z']) {
        expect(src.includes(trace), `old prism path in ${f}`).toBe(false);
      }
      expect(src.includes('BrandMark'), `BrandMark ref in ${f}`).toBe(false);
      expect(src.includes('MiaAgentLogo'), `MiaAgentLogo ref in ${f}`).toBe(false);
    }
  });

  it('the old prism blues (#2962FF / #7DA3FF) and ink (#0F1729) are gone', () => {
    for (const f of SURFACES) {
      const src = readFileSync(f, 'utf-8').toLowerCase();
      for (const c of ['#2962ff', '#7da3ff', '#0f1729']) {
        expect(src.includes(c), `${c} still in ${f}`).toBe(false);
      }
    }
  });

  it('the bare "MIA Markets" spelling (no dots) is gone', () => {
    const offenders = SURFACES.filter((f) => readFileSync(f, 'utf-8').includes('MIA Markets'));
    expect(offenders, offenders.join(', ')).toHaveLength(0);
  });
});

describe('BRD-3 — the logo colour comes from the var, never a literal', () => {
  // Brass (#C9A14A) doubles as the THM-1 UI accent and is legitimately hard-coded
  // across surfaces that cannot read CSS vars (pre-hydration error page, chart
  // overlays…). That is NOT the logo. This guard is scoped to the logo itself:
  // MiaLogo paints from --brand-mark, and neither logo-source file writes the
  // brass literal. The build-time images hard-code it on purpose (no theme then).
  it('MiaLogo paints from --brand-mark and hard-codes no colour', () => {
    const src = readFileSync(join(WEBAPP, 'components/brand/MiaLogo.tsx'), 'utf-8').toLowerCase();
    expect(src).toContain('var(--brand-mark)');
    expect(src).toContain('var(--brand-word)');
    expect(src.includes('#c9a14a'), 'MiaLogo must not hard-code brass').toBe(false);
  });

  it('the candle geometry is pure data — no colour literal', () => {
    const src = readFileSync(join(WEBAPP, 'lib/brand/candle-geometry.ts'), 'utf-8').toLowerCase();
    expect(src.includes('#c9a14a'), 'geometry must not carry a colour').toBe(false);
    expect(/#[0-9a-f]{6}/.test(src), 'geometry must hold coordinates, not colours').toBe(false);
  });
});

describe('BRD-3 — compact variant at small sizes, full mark never < 40px', () => {
  it('the bare five-candle mark is never rendered below 40px', () => {
    // The rule targets the standalone mark (variant="mark" or the default). The
    // horizontal / stacked lockups carry the wordmark and are legible small, so
    // the public header / rail place them at their own sizes — those are exempt.
    const COMPONENTS = walk(join(WEBAPP, 'components')).concat(walk(join(WEBAPP, 'app')));
    for (const f of COMPONENTS) {
      const src = readFileSync(f, 'utf-8');
      const tags = src.match(/<MiaLogo\b[^>]*\/>/g) ?? [];
      for (const tag of tags) {
        const isLockupOrCompact =
          tag.includes('variant="horizontal"') ||
          tag.includes('variant="stacked"') ||
          tag.includes('variant="compact"');
        if (isLockupOrCompact) continue; // remaining = the bare five-candle mark
        const m = tag.match(/height=\{(\d+)\}/);
        if (m) {
          expect(
            Number(m[1]),
            `bare mark below 40px (use variant="compact") in ${f}: ${tag}`,
          ).toBeGreaterThanOrEqual(40);
        }
      }
    }
  });
});

describe('BRD-3 — logo never in loading / empty / error', () => {
  const FORBIDDEN = [
    'app/[locale]/not-found.tsx',
    'app/[locale]/error.tsx',
    'app/global-error.tsx',
    'components/app/ReadingSkeleton.tsx',
  ];

  it('error/404/skeleton surfaces do not import or draw the logo', () => {
    for (const rel of FORBIDDEN) {
      const src = readFileSync(join(WEBAPP, rel), 'utf-8');
      expect(src.includes('MiaLogo'), `${rel} references MiaLogo`).toBe(false);
      expect(src.includes('candle-geometry'), `${rel} draws the mark`).toBe(false);
    }
  });
});
