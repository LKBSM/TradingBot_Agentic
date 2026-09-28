import { fireEvent, render, screen } from '@/components/test-utils';
import { describe, expect, it, vi } from 'vitest';
import fr from '@/messages/fr.json';
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
    // MARCHÉS — les deux instruments V1.
    //
    // Les libellés viennent de `calendar.market.*` depuis #240 (« les noms de
    // marchés suivent la langue de la page ») : une paire de change s'affiche
    // sous son code neutre (« EUR/USD »), un métal sous son nom traduit (« Or »
    // / « Gold »). Ce test codait « Euro / Dollar (EUR/USD) » en dur et est
    // devenu faux à cette fusion.
    //
    // Il lit donc maintenant la MÊME source que le composant. Ce qui est vérifié
    // ici, c'est que les deux marchés sont listés — pas l'orthographe du jour de
    // leur nom, qui appartient à la traduction.
    expect(screen.getByText(fr.calendar.market.XAUUSD)).toBeInTheDocument();
    expect(screen.getByText(fr.calendar.market.EURUSD)).toBeInTheDocument();
    // UNITÉ DE TEMPS — compact codes.
    expect(screen.getByText('M15')).toBeInTheDocument();
    expect(screen.getByText('H1')).toBeInTheDocument();
    expect(screen.getByText('H4')).toBeInTheDocument();
    // Educational disclaimer stays; the Freshbox live/instrument duplicate of the
    // AppHead header was removed (UI-3 text-density pass).
    expect(screen.queryByText('Lecture en direct')).not.toBeInTheDocument();
  });

  it('wires the ESPACE nav to the real routes', () => {
    render(<ShellRail activeSpace="zones" />);
    expect(screen.getByRole('link', { name: 'App' })).toHaveAttribute('href', '/app');
    // SC-2e: « Scanner » ouvre le mode conversationnel par défaut (palette à un
    // clic via la bascule). L'espace actif reste 'scanner' sur les deux routes.
    expect(screen.getByRole('link', { name: 'Scanner' })).toHaveAttribute(
      'href',
      '/scanner/decrire',
    );
    expect(screen.getByRole('link', { name: 'Zones' })).toHaveAttribute(
      'href',
      '/zones',
    );
    // "Réglages" points at the existing /compte route (nav label renamed UI-2b).
    expect(screen.getByRole('link', { name: 'Réglages' })).toHaveAttribute(
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
