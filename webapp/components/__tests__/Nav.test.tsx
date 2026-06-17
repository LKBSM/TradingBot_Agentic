import { fireEvent, render as rtlRender, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Nav } from '../Nav';
import { TooltipProvider } from '@/components/ui/tooltip';

// usePathname drives the marketing-vs-product header switch.
const hoisted = vi.hoisted(() => ({ pathname: '/' }));
vi.mock('next/navigation', () => ({
  usePathname: () => hoisted.pathname,
}));

// The real app wraps everything in a TooltipProvider (layout). LocaleToggle's
// radix Tooltip needs it.
function render(ui: React.ReactElement) {
  return rtlRender(<TooltipProvider>{ui}</TooltipProvider>);
}

afterEach(() => {
  hoisted.pathname = '/';
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

  it('keeps the marketing links inside the account menu (only on demand)', () => {
    hoisted.pathname = '/app';
    render(<Nav />);
    // Closed by default — site links not visible.
    expect(screen.queryByRole('menuitem', { name: /honnêteté/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /menu du compte/i }));

    const menu = screen.getByRole('menu', { name: /compte/i });
    expect(within(menu).getByRole('menuitem', { name: /abonnement/i })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: /honnêteté/i })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: /faq/i })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: /se déconnecter/i })).toBeInTheDocument();
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
