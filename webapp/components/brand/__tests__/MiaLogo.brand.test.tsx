/**
 * BRD-2 brand guards. Four properties the logo rollout must keep true:
 *  1. the prism geometry lives in exactly ONE code file (single source);
 *  2. MiaLogo renders the right variant + tone in light and dark;
 *  3. no old logo/name string survives on a user-facing surface;
 *  4. the logo never appears in a loading / empty / error state.
 */
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MiaLogo } from '../MiaLogo';

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

describe('BRD-2 — single logo source', () => {
  it('each prism path literal exists in exactly one code file', () => {
    for (const trace of ['M46,14 L78,82 L14,82 Z', 'M48,16 L78,84 L18,84 Z']) {
      const hits = CODE.filter((f) => readFileSync(f, 'utf-8').includes(trace));
      expect(hits, `trace ${trace} duplicated in: ${hits.join(', ')}`).toHaveLength(1);
      expect((hits[0] ?? '').replace(/\\/g, '/')).toContain('lib/brand/prism-geometry');
    }
  });

  it('MiaLogo has a single definition', () => {
    const defs = CODE.filter((f) => /export function MiaLogo\b/.test(readFileSync(f, 'utf-8')));
    expect(defs).toHaveLength(1);
  });

  it('no component redraws the old candlestick / gold-M marks', () => {
    for (const f of CODE) {
      const src = readFileSync(f, 'utf-8');
      // Old BrandMark candlestick rects + old gold-"M" favicon path.
      expect(src.includes('M120 384'), `old M path in ${f}`).toBe(false);
      expect(src.includes('BrandMark'), `BrandMark ref in ${f}`).toBe(false);
      expect(src.includes('MiaAgentLogo'), `MiaAgentLogo ref in ${f}`).toBe(false);
    }
  });
});

describe('BRD-2 — variant + tone', () => {
  it('renders the labelled prism by default (role img)', () => {
    const { getByRole } = render(<MiaLogo />);
    const svg = getByRole('img');
    expect(svg.getAttribute('aria-label')).toBe('M.I.A Markets');
  });

  it('is hidden from screen readers when decorative', () => {
    const { container, queryByRole } = render(<MiaLogo decorative />);
    expect(queryByRole('img')).toBeNull();
    expect(container.querySelector('svg')?.getAttribute('aria-hidden')).toBe('true');
  });

  it('the horizontal + stacked lockups spell the company name with dots', () => {
    for (const variant of ['horizontal', 'stacked'] as const) {
      const { container } = render(<MiaLogo variant={variant} />);
      expect(container.textContent).toContain('M.I.A Markets');
    }
  });

  it('auto tone follows the theme variable; fixed tones use the source colours', () => {
    const fill = (ui: React.ReactElement) =>
      render(ui).container.querySelector('path')?.getAttribute('fill');
    // Auto → CSS var that resolves to #2962FF (light) / #7DA3FF (dark).
    expect(fill(<MiaLogo tone="auto" />)).toContain('--brand-mark');
    expect(fill(<MiaLogo tone="color" />)).toBe('#2962FF');
    expect(fill(<MiaLogo tone="dark" />)).toBe('#7DA3FF');
    expect(fill(<MiaLogo tone="mono" />)).toBe('currentColor');
  });
});

describe('BRD-2 — no old brand string on user-facing surfaces', () => {
  const SURFACES = walk(join(WEBAPP, 'components'))
    .concat(walk(join(WEBAPP, 'app')))
    .concat(walk(join(WEBAPP, 'messages'), ['.json']));

  it('the bare "MIA Markets" spelling (no dots) is gone', () => {
    // "M.I.A Markets" does not contain the substring "MIA Markets", so a plain
    // includes() flags only the old spelling.
    const offenders = SURFACES.filter((f) => readFileSync(f, 'utf-8').includes('MIA Markets'));
    expect(offenders, offenders.join(', ')).toHaveLength(0);
  });

  it('the old gold-gradient favicon colours are gone from the icon generators', () => {
    for (const f of ['app/icon.tsx', 'app/apple-icon.tsx', 'app/opengraph-image.tsx']) {
      const src = readFileSync(join(WEBAPP, f), 'utf-8');
      expect(src.includes('#FBBF24'), `gold gradient still in ${f}`).toBe(false);
      expect(src.includes('#B45309'), `gold gradient still in ${f}`).toBe(false);
    }
  });
});

describe('BRD-2 — logo never in loading / empty / error', () => {
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
      expect(src.includes('prism-geometry'), `${rel} draws the prism`).toBe(false);
    }
  });
});
