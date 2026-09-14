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

/** The pin toggle of ONE market, by its exact label. */
function pinButton(label: string) {
  return screen.getByLabelText(new RegExp(`^Épingler ${escapeRe(label)}$`, 'i'));
}

/**
 * Pin every market the selector renders. DATA-4 took the registry from 2 markets
 * to 80, so a test about "all pinned" can no longer name them one by one — it has
 * to derive the set, or it silently stops testing what it claims to.
 */
function pinEveryMarket() {
  for (const spec of MARKET_SPECS) {
    const btn = screen.queryByLabelText(new RegExp(`^Épingler ${escapeRe(spec.label)}$`, 'i'));
    if (btn) fireEvent.click(btn);
  }
}

/**
 * Market labels rendered more than once — the anti-duplication guard, counted in
 * a SINGLE pass. One `getAllByText` per market is 80 DOM scans and turns the
 * guard into a 30 s test without covering anything more.
 */
function duplicatedLabels({ allowTwice = [] as string[] } = {}) {
  const text = document.body.textContent ?? '';
  return MARKET_SPECS.filter((spec) => {
    const occurrences = text.split(spec.label).length - 1;
    const allowed = allowTwice.includes(spec.id) ? 2 : 1;
    return occurrences !== allowed;
  }).map((spec) => `${spec.id} x${text.split(spec.label).length - 1}`);
}

function escapeRe(text: string) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

describe('MarketSelector — registry is the single source (panel)', () => {
  it('lists exactly the registry markets, no phantom', () => {
    const { container } = render(
      <MarketSelector variant="panel" active={active} onSelect={() => {}} />,
    );
    // ONE pass over the rendered text rather than one DOM scan per market: at 80
    // markets the per-market query turned this into a 6 s test for no extra
    // coverage.
    const rendered = container.textContent ?? '';
    const missing = MARKET_SPECS.filter((spec) => !rendered.includes(spec.label));
    expect(missing.map((s) => s.id)).toEqual([]);
    // A market absent from the registry must never appear. DATA-4 made BTC a
    // followed market, so the example is now an index — catalogue-only by design.
    expect(rendered).not.toMatch(/S&P 500/i);
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
    const pinBtn = pinButton('Euro / Dollar (EUR/USD)');
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

  // Pinning every market means one re-render of an 80-row list per click, so this
  // one is legitimately slow — it exercises the whole catalogue on purpose.
  it('every market pinned → « Marchés » does not repeat them (APP-1 défaut B)', () => {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    // Pin EVERY market. Once all are pinned, the « Marchés » section has nothing
    // NEW to show and must not repeat the pinned list. Derived from the registry
    // so this keeps holding as markets are added (DATA-4 took it from 2 to 80).
    pinEveryMarket();
    // Each market now appears EXACTLY once (in « Épinglés »), never duplicated.
    expect(duplicatedLabels()).toEqual([]);
    // « Épinglés » stays; the redundant « Marchés » heading is gone.
    expect(screen.getByText('Épinglés')).toBeTruthy();
    expect(screen.queryByText('Marchés')).toBeNull();
  }, 60000);
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

  // ── VZ-5 — anti-duplication, the two paths the founder reported ───────────
  // The column form has had this guard since APP-1; the bar form never did, and
  // that is exactly where the omission survived. Both paths are locked here.

  it('VZ-5 — pinning from the dropdown lists the market ONCE, not twice', () => {
    render(<MarketSelector variant="bar" active={active} onSelect={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /Marchés/i }));
    // EURUSD (not the active market, so the closed trigger never counts).
    expect(screen.getAllByText('Euro / Dollar (EUR/USD)')).toHaveLength(1);

    fireEvent.click(pinButton('Euro / Dollar (EUR/USD)'));

    // It moved INTO « Épinglés » — it did not get added on top of the full list.
    expect(screen.getByText('Épinglés')).toBeTruthy();
    expect(screen.getAllByText('Euro / Dollar (EUR/USD)')).toHaveLength(1);
  });

  it('VZ-5 — searching then pinning from the result lists the market ONCE', () => {
    render(<MarketSelector variant="bar" active={active} onSelect={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /Marchés/i }));
    fireEvent.change(screen.getByLabelText(/Rechercher un marché/i), {
      target: { value: 'euro' },
    });
    expect(screen.getAllByText('Euro / Dollar (EUR/USD)')).toHaveLength(1);

    // Pin straight from the filtered result — the path that made the duplicate
    // appear under the user's eyes, the search field being right above it.
    fireEvent.click(pinButton('Euro / Dollar (EUR/USD)'));

    expect(screen.getAllByText('Euro / Dollar (EUR/USD)')).toHaveLength(1);
  });

  it('VZ-5 — every market pinned → the bar form repeats nothing either', () => {
    render(<MarketSelector variant="bar" active={active} onSelect={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /Marchés/i }));
    pinEveryMarket();

    // The ACTIVE market's label is also the trigger's, so it legitimately shows
    // twice; every other market must appear exactly once.
    expect(duplicatedLabels({ allowTwice: [active.instrument] })).toEqual([]);
  }, 60000);

  it('VZ-5 — a search matching nothing still states it explicitly', () => {
    render(<MarketSelector variant="bar" active={active} onSelect={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /Marchés/i }));
    fireEvent.change(screen.getByLabelText(/Rechercher un marché/i), {
      target: { value: 'zzzz' },
    });
    expect(screen.getByText(/Aucun marché ne correspond/i)).toBeTruthy();
  });
});
