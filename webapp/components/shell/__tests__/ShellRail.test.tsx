import { fireEvent, render, screen } from '@/components/test-utils';
import { describe, expect, it, vi } from 'vitest';
import { ShellRail } from '../ShellRail';

// The rail reads/writes the active combo through the router + URL (single source
// of truth). Stub the app-router hooks: on /app, empty query → default XAU M15.
const { push, replace } = vi.hoisted(() => ({ push: vi.fn(), replace: vi.fn() }));
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push, replace }),
  usePathname: () => '/app',
  useSearchParams: () => new URLSearchParams(),
}));

describe('ShellRail', () => {
  it('renders the four sections from the reference', () => {
    render(<ShellRail activeSpace="app" />);
    // Market search.
    expect(
      screen.getByPlaceholderText(/rechercher un marché/i),
    ).toBeInTheDocument();
    // MARCHÉS — both V1 instruments.
    expect(screen.getByText('Or (XAU/USD)')).toBeInTheDocument();
    expect(screen.getByText('Euro / Dollar (EUR/USD)')).toBeInTheDocument();
    // UNITÉ DE TEMPS — compact codes.
    expect(screen.getByText('M15')).toBeInTheDocument();
    expect(screen.getByText('H1')).toBeInTheDocument();
    expect(screen.getByText('H4')).toBeInTheDocument();
    // Freshbox microcopy.
    expect(screen.getByText('Lecture en direct')).toBeInTheDocument();
  });

  it('wires the ESPACE nav to the real routes', () => {
    render(<ShellRail activeSpace="zones" />);
    expect(screen.getByRole('link', { name: 'App' })).toHaveAttribute('href', '/app');
    expect(screen.getByRole('link', { name: 'Scanner' })).toHaveAttribute(
      'href',
      '/scanner',
    );
    expect(screen.getByRole('link', { name: 'Zones' })).toHaveAttribute(
      'href',
      '/zones',
    );
    // "Réglages" points at the existing /compte route (decision UI-1).
    expect(screen.getByRole('link', { name: 'Compte' })).toHaveAttribute(
      'href',
      '/compte',
    );
    // The active space is marked current.
    expect(screen.getByRole('link', { name: 'Zones' })).toHaveAttribute(
      'aria-current',
      'page',
    );
  });

  it('writes the chosen timeframe into the URL (source of truth)', () => {
    replace.mockClear();
    render(<ShellRail activeSpace="app" />);
    fireEvent.click(screen.getByText('H1'));
    // On /app → replace (no history spam); default instrument XAUUSD is kept.
    expect(replace).toHaveBeenCalledWith(
      '/app?instrument=XAUUSD&timeframe=H1',
      { scroll: false },
    );
  });
});
