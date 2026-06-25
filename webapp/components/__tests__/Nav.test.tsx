import { fireEvent, render as rtlRender, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Nav } from '../Nav';
import { TooltipProvider } from '@/components/ui/tooltip';
import { AuthProvider } from '@/lib/auth/store';

// usePathname drives the marketing-vs-product header switch; useRouter is used
// by the (now session-aware) AccountMenu.
const hoisted = vi.hoisted(() => ({ pathname: '/' }));
vi.mock('next/navigation', () => ({
  usePathname: () => hoisted.pathname,
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

// AuthProvider probes /api/auth/me on mount. Stub fetch → 401 so the tree
// renders in a deterministic logged-out state without any real network.
beforeEach(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      new Response(JSON.stringify({ detail: 'Authentication required' }), {
        status: 401,
        headers: { 'content-type': 'application/json' },
      }),
    ),
  );
});

// The real app wraps everything in a TooltipProvider + AuthProvider (layout).
function render(ui: React.ReactElement) {
  return rtlRender(
    <TooltipProvider>
      <AuthProvider>{ui}</AuthProvider>
    </TooltipProvider>,
  );
}

afterEach(() => {
  hoisted.pathname = '/';
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('Nav — marketing landing', () => {
  it('shows the marketing section anchors', () => {
    hoisted.pathname = '/';
    render(<Nav />);
    const nav = screen.getByRole('navigation', { name: /sections du site/i });
    expect(within(nav).getByText('Démo')).toBeInTheDocument();
    expect(within(nav).getByText('Honnêteté')).toBeInTheDocument();
    expect(within(nav).getByText('Tarifs')).toBeInTheDocument();
    expect(within(nav).getByText('FAQ')).toBeInTheDocument();
  });

  it('exposes a session-aware account control (Connexion when logged out)', async () => {
    hoisted.pathname = '/';
    render(<Nav />);
    await waitFor(() =>
      expect(screen.getByRole('link', { name: /connexion/i })).toBeInTheDocument(),
    );
  });
});

describe('Nav — /app product header', () => {
  it('swaps to the product header: brand + utility cluster, NO marketing nav', () => {
    hoisted.pathname = '/app';
    render(<Nav />);
    // No marketing section nav at all.
    expect(
      screen.queryByRole('navigation', { name: /sections du site/i }),
    ).not.toBeInTheDocument();
    // No marketing anchors surfaced in the header itself.
    expect(screen.queryByText('Démo')).not.toBeInTheDocument();
    expect(screen.queryByText('Tarifs')).not.toBeInTheDocument();
    // Brand + utility controls present.
    expect(screen.getByRole('link', { name: /espace de lecture/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /menu du compte/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /aide/i })).toBeInTheDocument();
  });

  it('account menu shows auth entries (logged out) + marketing links on demand', () => {
    hoisted.pathname = '/app';
    render(<Nav />);
    // Closed by default — items not visible.
    expect(screen.queryByRole('menuitem', { name: /honnêteté/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /menu du compte/i }));

    const menu = screen.getByRole('menu', { name: /compte/i });
    // Logged out → Connexion + Inscription, NOT logout.
    expect(within(menu).getByRole('menuitem', { name: /se connecter/i })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: /créer un compte/i })).toBeInTheDocument();
    expect(within(menu).queryByRole('menuitem', { name: /se déconnecter/i })).not.toBeInTheDocument();
    // Marketing links remain reachable from the menu.
    expect(within(menu).getByRole('menuitem', { name: /honnêteté/i })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: /faq/i })).toBeInTheDocument();
  });

  it('resolves /app even with a defensive locale prefix', () => {
    hoisted.pathname = '/fr/app';
    render(<Nav />);
    expect(
      screen.queryByRole('navigation', { name: /sections du site/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /menu du compte/i })).toBeInTheDocument();
  });
});
