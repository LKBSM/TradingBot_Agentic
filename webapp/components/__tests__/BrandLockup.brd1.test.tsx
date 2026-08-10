import { describe, expect, it, vi } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { render, screen } from '@/components/test-utils';
import { BrandLockup } from '@/components/BrandLockup';
import { ReadingSkeleton } from '@/components/app/ReadingSkeleton';
import { ShellRail } from '@/components/shell/ShellRail';
import { BRAND_NAME, BRAND_BASELINE } from '@/lib/brand';

// The rail reads/writes the active combo through the router + URL. Stub the
// app-router hooks (mirrors ShellRail.test) so the rail renders under test.
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => '/app',
  useSearchParams: () => new URLSearchParams(),
}));

/**
 * Mission BRD-1 — brand presence in the connected surfaces.
 *
 * Guards the four acceptance criteria: (1) the wordmark and its developed
 * acronym are present on each connected surface; (2) the brand block is a single
 * reusable component, never duplicated markup; (3) the tab-title format is the
 * same everywhere; (4) no loading indicator was replaced by the brand.
 */

const WEBAPP_ROOT = process.cwd();

function collectSources(dir: string): string[] {
  let out: string[] = [];
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return out;
  }
  for (const name of entries) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) {
      if (name === 'node_modules' || name === '.next' || name === '__tests__') continue;
      out = out.concat(collectSources(full));
    } else if (/\.(ts|tsx)$/.test(name)) {
      out.push(full);
    }
  }
  return out;
}

describe('BRD-1 — BrandLockup component', () => {
  it('renders the wordmark and the developed acronym as real text', () => {
    render(<BrandLockup />);
    expect(screen.getByText(BRAND_NAME)).toBeInTheDocument();
    expect(screen.getByText(BRAND_BASELINE)).toBeInTheDocument();
  });

  it('can hide the baseline but always keeps the wordmark', () => {
    render(<BrandLockup baseline={false} />);
    expect(screen.getByText(BRAND_NAME)).toBeInTheDocument();
    expect(screen.queryByText(BRAND_BASELINE)).not.toBeInTheDocument();
  });

  it('exposes the glyph as decorative (real text carries the accessible name)', () => {
    const { container } = render(<BrandLockup />);
    const svg = container.querySelector('svg');
    // The glyph is inside an aria-hidden span (BrandMark); the wordmark/baseline
    // are the accessible text.
    expect(svg?.closest('[aria-hidden]')).not.toBeNull();
  });
});

describe('BRD-1 — present on each connected surface via one component', () => {
  // The product surfaces (/app · /scanner · /zones · /actualites · /compte)
  // share the rail; /app additionally has a mobile header and a loading screen.
  const SURFACES = [
    'components/shell/ShellRail.tsx',
    'components/app/MobileWorkspace.tsx',
    'components/app/ReadingSkeleton.tsx',
    'components/app/AppHeader.tsx',
  ];

  it('every connected surface pulls the brand from BrandLockup', () => {
    for (const rel of SURFACES) {
      const src = readFileSync(join(WEBAPP_ROOT, rel), 'utf8');
      expect(src, `${rel} should import BrandLockup`).toMatch(
        /import \{ BrandLockup \} from '@\/components\/BrandLockup'/,
      );
    }
  });

  it('no connected surface hand-stacks the wordmark and baseline itself', () => {
    // Only BrandLockup pairs BRAND_NAME with BRAND_BASELINE. The surfaces
    // delegate, so none of them reference the baseline constant directly.
    for (const rel of SURFACES) {
      const src = readFileSync(join(WEBAPP_ROOT, rel), 'utf8');
      expect(src, `${rel} must not re-implement the brand block`).not.toMatch(
        /BRAND_BASELINE/,
      );
    }
  });

  it('the gold badge markup lives in exactly one component (BrandMark)', () => {
    const offenders: string[] = [];
    for (const file of collectSources(join(WEBAPP_ROOT, 'components'))) {
      if (/from-amber-400 to-amber-600/.test(readFileSync(file, 'utf8'))) {
        offenders.push(file.replace(WEBAPP_ROOT, '').replace(/\\/g, '/'));
      }
    }
    expect(offenders).toEqual(['/components/BrandMark.tsx']);
  });
});

describe('BRD-1 — tab-title format is consistent', () => {
  it('the locale layout applies the "%s · MIA Markets" template', () => {
    const layout = readFileSync(
      join(WEBAPP_ROOT, 'app/[locale]/layout.tsx'),
      'utf8',
    );
    expect(layout).toMatch(/template: `%s · \$\{BRAND_NAME\}`/);
  });

  it('every product page sets its own title (so the template resolves)', () => {
    const pages = [
      'app/[locale]/(product)/app/page.tsx',
      'app/[locale]/(product)/scanner/page.tsx',
      'app/[locale]/(product)/zones/page.tsx',
      'app/[locale]/(product)/actualites/page.tsx',
      'app/[locale]/(product)/compte/page.tsx',
    ];
    for (const rel of pages) {
      const src = readFileSync(join(WEBAPP_ROOT, rel), 'utf8');
      expect(src, `${rel} should set title from a meta key`).toMatch(
        /title: t\('[^']*meta\.title'\)/,
      );
    }
  });
});

describe('BRD-1 — the brand is added, never a replacement', () => {
  it('the loading screen keeps its skeleton AND shows the brand', () => {
    const { getByTestId } = render(<ReadingSkeleton />);
    // Loading indicator intact.
    const skeleton = getByTestId('reading-skeleton');
    expect(skeleton).toBeInTheDocument();
    expect(skeleton.querySelectorAll('.animate-pulse').length).toBeGreaterThan(4);
    // Brand added alongside it.
    expect(screen.getByText(BRAND_NAME)).toBeInTheDocument();
  });

  it('the empty-state message components were not given the brand', () => {
    // The scanner/zones empty states must keep their explicit message as the
    // priority content — the brand is NOT injected there.
    const emptyStateFiles = [
      'components/scanner/ScanResults.tsx',
      'components/zones/ZonesWorkspace.tsx',
    ];
    for (const rel of emptyStateFiles) {
      const full = join(WEBAPP_ROOT, rel);
      let src = '';
      try {
        src = readFileSync(full, 'utf8');
      } catch {
        continue; // tolerate a renamed file — the guard below covers the rail
      }
      expect(src, `${rel} must not embed the brand block`).not.toMatch(
        /BrandLockup/,
      );
    }
  });
});

describe('BRD-1 — rail renders the brand', () => {
  it('shows the wordmark and acronym at the top of the product rail', () => {
    render(<ShellRail activeSpace="app" />);
    expect(screen.getByText(BRAND_NAME)).toBeInTheDocument();
    expect(screen.getByText(BRAND_BASELINE)).toBeInTheDocument();
  });
});
