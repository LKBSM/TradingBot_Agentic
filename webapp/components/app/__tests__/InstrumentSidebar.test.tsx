import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { InstrumentSidebar } from '../InstrumentSidebar';
import { SUPPORTED_COMBOS, type Combo } from '@/lib/market-reading/store';

function renderSidebar(active: Combo | null = null) {
  const onSelect = vi.fn();
  render(
    <InstrumentSidebar
      combos={SUPPORTED_COMBOS}
      active={active}
      onSelect={onSelect}
      activeCandleCloseTs={null}
    />,
  );
  return { onSelect };
}

beforeEach(() => {
  window.localStorage.clear();
});

describe('InstrumentSidebar — catalogue + search filter', () => {
  it('lists both instruments and their 3 timeframes (6 select buttons)', () => {
    renderSidebar();
    expect(screen.getByText('Or (XAU/USD)')).toBeInTheDocument();
    expect(screen.getByText('Euro / Dollar (EUR/USD)')).toBeInTheDocument();
    // 6 select buttons + 6 pin buttons = 12 buttons total.
    expect(screen.getAllByRole('button', { name: /Épingler/ })).toHaveLength(6);
  });

  it('filters the catalogue by instrument name', () => {
    renderSidebar();
    const search = screen.getByRole('searchbox');
    fireEvent.change(search, { target: { value: 'euro' } });

    expect(screen.getByText('Euro / Dollar (EUR/USD)')).toBeInTheDocument();
    expect(screen.queryByText('Or (XAU/USD)')).not.toBeInTheDocument();
  });

  it('filters by timeframe across both instruments', () => {
    renderSidebar();
    const search = screen.getByRole('searchbox');
    fireEvent.change(search, { target: { value: 'H4' } });

    // Both instruments still shown, but only their H4 row → 2 pin buttons.
    expect(screen.getByText('Or (XAU/USD)')).toBeInTheDocument();
    expect(screen.getByText('Euro / Dollar (EUR/USD)')).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /Épingler/ })).toHaveLength(2);
  });

  it('shows an empty-result message when nothing matches', () => {
    renderSidebar();
    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: 'zzz' },
    });
    expect(screen.getByText(/Aucun marché ne correspond/)).toBeInTheDocument();
  });

  it('only ever exposes the fixed 6-combo catalogue (no external symbols)', () => {
    renderSidebar();
    // No instrument outside the V1 perimeter leaks into the list.
    for (const code of ['BTC', 'US500', 'GBP', 'JPY']) {
      expect(screen.queryByText(new RegExp(code, 'i'))).not.toBeInTheDocument();
    }
  });
});

describe('InstrumentSidebar — pin', () => {
  it('pins a combo and floats it into the "Épinglés" section', () => {
    renderSidebar();
    const pinBtn = screen.getAllByRole('button', { name: /^Épingler/ })[0]!;
    fireEvent.click(pinBtn);

    const pinnedHeading = screen.getByText('Épinglés');
    expect(pinnedHeading).toBeInTheDocument();
    // Persisted.
    expect(window.localStorage.getItem('mia.pinnedCombos.v1')).toContain(
      'XAUUSD',
    );
  });

  it('unpins via the toggle', () => {
    renderSidebar();
    fireEvent.click(screen.getAllByRole('button', { name: /^Épingler/ })[0]!);
    expect(screen.getByText('Épinglés')).toBeInTheDocument();

    // Now a "Désépingler" toggle exists; clicking it removes the section.
    fireEvent.click(screen.getAllByRole('button', { name: /^Désépingler/ })[0]!);
    expect(screen.queryByText('Épinglés')).not.toBeInTheDocument();
  });
});

describe('InstrumentSidebar — active state', () => {
  it('marks the active combo with aria-current', () => {
    renderSidebar({ instrument: 'XAUUSD', timeframe: 'M15' });
    const current = screen
      .getAllByRole('button')
      .filter((b) => b.getAttribute('aria-current') === 'true');
    expect(current).toHaveLength(1);
  });
});
