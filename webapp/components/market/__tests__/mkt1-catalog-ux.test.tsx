import { render as rtlRender, screen, fireEvent, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import messages from '@/messages/fr.json';

/**
 * MKT-1 test UX — the selector WITH the catalogue on (100 entries).
 *
 * The flag itself is inlined at build time; whether it opens is covered by
 * lib/__tests__/market-catalog-flag.test.ts. Here we force it on and test what
 * the component does at that scale. Only the gate is mocked — the catalogue data
 * is the real generated one, so these assertions run against the actual 100.
 */
// The flag is read at module load, so it is stubbed BEFORE the dynamic imports
// below. Nothing is mocked: this exercises the real gate and the real catalogue.
vi.stubEnv('NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST', '1');

const { MarketSelector } = await import('../MarketSelector');
const { CATALOG_ONLY_COUNT, CATALOG_ONLY_ENTRIES } = await import('@/lib/market-catalog');
const { MARKET_SPECS } = await import('@/lib/markets');

function render(ui: React.ReactElement) {
  return rtlRender(
    <NextIntlClientProvider locale="fr" messages={messages}>
      {ui}
    </NextIntlClientProvider>,
  );
}

const active = { instrument: 'XAUUSD', timeframe: 'M15' };
const GROUPS = ['fx-major', 'fx-minor', 'fx-exotic', 'metal', 'index', 'crypto'];

beforeEach(() => {
  window.localStorage.clear();
});

describe('MKT-1 — catalogue on: the mode is visible and the scale is grouped', () => {
  it('shows a permanent « Mode test UX » banner with the real ratio', () => {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    const banner = screen.getByTestId('mkt-uxtest-banner');
    expect(banner).toHaveTextContent('Mode test UX');
    // The counts are computed, so the banner cannot drift from the catalogue.
    expect(banner).toHaveTextContent(String(CATALOG_ONLY_COUNT));
    expect(banner).toHaveTextContent(String(MARKET_SPECS.length));
  });

  it('separates the markets the engine follows from the display-only ones', () => {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    expect(screen.getByText('Suivis par le moteur')).toBeInTheDocument();
    for (const spec of MARKET_SPECS) {
      expect(screen.getAllByText(spec.label).length).toBeGreaterThan(0);
    }
  });

  it('renders the six categories, every one collapsed at first', () => {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    for (const group of GROUPS) {
      const section = screen.getByTestId(`mkt-group-${group}`);
      const head = screen.getByTestId(`mkt-group-head-${group}`);
      expect(head, `${group} should start collapsed`).toHaveAttribute('aria-expanded', 'false');
      // Collapsed means the rows are not in the DOM at all.
      expect(within(section).queryAllByRole('listitem')).toHaveLength(0);
    }
  });

  it('a category header states how many markets it holds, and opens on click', () => {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    const section = screen.getByTestId('mkt-group-metal');
    const head = screen.getByTestId('mkt-group-head-metal');
    const metals = CATALOG_ONLY_ENTRIES.filter((e) => e.group === 'metal');
    expect(head).toHaveTextContent(String(metals.length));

    fireEvent.click(head);
    expect(head).toHaveAttribute('aria-expanded', 'true');
    expect(within(section).getAllByRole('listitem')).toHaveLength(metals.length);
    expect(within(section).getByText('Argent (XAG/USD)')).toBeInTheDocument();
  });
});

describe('MKT-1 — search at 100 entries', () => {
  function searchFor(query: string) {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    fireEvent.change(screen.getByLabelText(/Rechercher un marché/i), { target: { value: query } });
  }

  it('finds a catalogue market by id, and opens its category so the hit is visible', () => {
    searchFor('XAGUSD');
    const section = screen.getByTestId('mkt-group-metal');
    expect(screen.getByTestId('mkt-group-head-metal')).toHaveAttribute('aria-expanded', 'true');
    expect(within(section).getByText('Argent (XAG/USD)')).toBeInTheDocument();
  });

  it('finds a market by its human label', () => {
    searchFor('Bitcoin');
    const section = screen.getByTestId('mkt-group-crypto');
    expect(within(section).getByText('Bitcoin (BTC/USD)')).toBeInTheDocument();
  });

  it('finds markets by CATEGORY name — how one actually looks at this scale', () => {
    searchFor('indices');
    const section = screen.getByTestId('mkt-group-index');
    const indices = CATALOG_ONLY_ENTRIES.filter((e) => e.group === 'index');
    expect(within(section).getAllByRole('listitem')).toHaveLength(indices.length);
    // A category search must not drag in unrelated categories.
    expect(screen.queryByTestId('mkt-group-crypto')).not.toBeInTheDocument();
  });

  it('a query matching nothing says so explicitly — never a silent fallback', () => {
    searchFor('zzzzz');
    expect(screen.getByText(/Aucun marché ne correspond/)).toBeInTheDocument();
    for (const group of GROUPS) {
      expect(screen.queryByTestId(`mkt-group-${group}`)).not.toBeInTheDocument();
    }
  });
});

describe('MKT-1 — performance of the 100-entry list', () => {
  it('mounts and filters without a noticeable slowdown', () => {
    const t0 = performance.now();
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    const mount = performance.now() - t0;

    const input = screen.getByLabelText(/Rechercher un marché/i);
    // Type a query character by character — the filter runs on every keystroke
    // over the whole catalogue, which is the cost this test exists to bound.
    const t1 = performance.now();
    const query = 'bitcoin';
    for (let i = 1; i <= query.length; i += 1) {
      fireEvent.change(input, { target: { value: query.slice(0, i) } });
    }
    const typing = performance.now() - t1;

    // Deliberately generous bounds: this guards against an accidental O(n²) or a
    // dropped memo, not against a slow CI machine. Typical local run is an order
    // of magnitude under.
    expect(mount, `mount took ${mount.toFixed(0)}ms`).toBeLessThan(3000);
    expect(typing, `${query.length} keystrokes took ${typing.toFixed(0)}ms`).toBeLessThan(3000);
  });

  it('keeps the DOM proportional to what is OPEN, not to the catalogue size', () => {
    const { container } = render(
      <MarketSelector variant="panel" active={active} onSelect={() => {}} />,
    );
    // Collapsed by default: ~the real markets only, not 98 extra rows. This is
    // why no virtualisation is needed — the wall never renders in the first place.
    const rows = container.querySelectorAll('[data-catalog-only="true"]');
    expect(rows).toHaveLength(0);

    fireEvent.click(screen.getByTestId('mkt-group-head-metal'));
    const metals = CATALOG_ONLY_ENTRIES.filter((e) => e.group === 'metal').length;
    expect(container.querySelectorAll('[data-catalog-only="true"]')).toHaveLength(metals);
  });
});

describe('MKT-1 — a display-only market promises nothing', () => {
  it('offers no pin affordance (pinning a market with no data would be a promise)', () => {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    const section = screen.getByTestId('mkt-group-crypto');
    fireEvent.click(screen.getByTestId('mkt-group-head-crypto'));
    expect(within(section).queryByRole('button', { name: /Épingler/i })).not.toBeInTheDocument();
  });

  it('picking one emits its combo, keeping the timeframe the user was on', () => {
    const onSelect = vi.fn();
    render(<MarketSelector variant="panel" active={active} onSelect={onSelect} />);
    fireEvent.click(screen.getByTestId('mkt-group-head-crypto'));
    fireEvent.click(screen.getByText('Bitcoin (BTC/USD)'));
    expect(onSelect).toHaveBeenCalledWith({ instrument: 'BTCUSD', timeframe: 'M15' });
  });
});
