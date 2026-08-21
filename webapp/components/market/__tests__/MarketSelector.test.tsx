import { render as rtlRender, screen, fireEvent, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import { MarketSelector } from '../MarketSelector';
import { MARKET_SPECS } from '@/lib/markets';
import messages from '@/messages/fr.json';

function render(ui: React.ReactElement) {
  return rtlRender(
    <NextIntlClientProvider locale="fr" messages={messages}>
      {ui}
    </NextIntlClientProvider>,
  );
}

const active = { instrument: 'XAUUSD', timeframe: 'M15' };

beforeEach(() => {
  window.localStorage.clear();
});

describe('MarketSelector — registry is the single source (panel)', () => {
  it('lists exactly the registry markets, no phantom', () => {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    for (const spec of MARKET_SPECS) {
      expect(screen.getAllByText(spec.label).length).toBeGreaterThan(0);
    }
    // A market absent from the registry must never appear.
    expect(screen.queryByText(/Bitcoin/i)).toBeNull();
  });

  it('search with no match shows an explicit message, never a silent fallback', () => {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    const search = screen.getByLabelText(/Rechercher un marché/i);
    fireEvent.change(search, { target: { value: 'zzzz-nothing' } });
    expect(screen.getByText(/Aucun marché ne correspond/i)).toBeTruthy();
  });

  it('search filters to the matching market only', () => {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    const search = screen.getByLabelText(/Rechercher un marché/i);
    fireEvent.change(search, { target: { value: 'euro' } });
    expect(screen.getByText('Euro / Dollar (EUR/USD)')).toBeTruthy();
    expect(screen.queryByText('Or (XAU/USD)')).toBeNull();
  });

  it('selecting a market emits its combo (keeps the current timeframe)', () => {
    const onSelect = vi.fn();
    render(<MarketSelector variant="panel" active={active} onSelect={onSelect} />);
    fireEvent.click(screen.getByText('Euro / Dollar (EUR/USD)'));
    expect(onSelect).toHaveBeenCalledWith({ instrument: 'EURUSD', timeframe: 'M15' });
  });

  it('pin a market → it surfaces in the "Épinglés" section and persists', () => {
    const { unmount } = render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    // Pin EURUSD via its pin toggle.
    const pinBtn = screen.getByLabelText(/Épingler Euro \/ Dollar/i);
    fireEvent.click(pinBtn);
    expect(window.localStorage.getItem('mia.pinnedMarkets.v1')).toContain('EURUSD');
    unmount();
    // Remount: the pinned section renders the market at the top.
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    const pinnedHeading = screen.getByText('Épinglés');
    expect(pinnedHeading).toBeTruthy();
    // The "not synced" mention is shown next to the pinned heading.
    expect(screen.getAllByText('Non synchronisé').length).toBeGreaterThan(0);
  });
});

describe('MarketSelector — timeframe control', () => {
  it('renders the active market timeframes and emits a combo on pick', () => {
    const onSelect = vi.fn();
    render(<MarketSelector variant="panel" active={active} onSelect={onSelect} />);
    const h1 = screen.getByRole('button', { name: '1 heure' });
    fireEvent.click(h1);
    expect(onSelect).toHaveBeenCalledWith({ instrument: 'XAUUSD', timeframe: 'H1' });
  });
});

describe('MarketSelector — bar (header) variant', () => {
  it('opens a searchable dropdown and picks a market', () => {
    const onSelect = vi.fn();
    render(<MarketSelector variant="bar" active={active} onSelect={onSelect} />);
    // The trigger shows the active market.
    const trigger = screen.getByRole('button', { name: /Marchés/i });
    fireEvent.click(trigger);
    const search = screen.getByLabelText(/Rechercher un marché/i);
    fireEvent.change(search, { target: { value: 'euro' } });
    const list = screen.getByRole('list');
    fireEvent.click(within(list).getByText('Euro / Dollar (EUR/USD)'));
    expect(onSelect).toHaveBeenCalledWith({ instrument: 'EURUSD', timeframe: 'M15' });
  });
});
