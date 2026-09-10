import { fireEvent, render as rtlRender, renderHook, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import { MarketSelector } from '../MarketSelector';
import { useMarketReading } from '@/lib/market-reading/hooks';
import { MARKET_SPECS } from '@/lib/markets';
import { CATALOG_UX_TEST_ENABLED } from '@/lib/market-catalog';
import messages from '@/messages/fr.json';

/**
 * MKT-1 — the default state of the product.
 *
 * Nothing is mocked here on purpose: this file runs with the flag exactly as a
 * standard build has it (unset), and asserts that the whole feature is absent
 * and the selector behaves as it did before the mission.
 */

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

describe('MKT-1 — with no env var set, the UX-test catalogue does not exist', () => {
  it('the flag is off (this is what a standard production build compiles)', () => {
    expect(process.env.NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST).toBeFalsy();
    expect(CATALOG_UX_TEST_ENABLED).toBe(false);
  });

  it('no « Mode test UX » banner anywhere', () => {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    expect(screen.queryByTestId('mkt-uxtest-banner')).not.toBeInTheDocument();
    expect(screen.queryByText(/Mode test UX/)).not.toBeInTheDocument();
  });

  it('no category section', () => {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    for (const group of GROUPS) {
      expect(screen.queryByTestId(`mkt-group-${group}`)).not.toBeInTheDocument();
    }
  });

  it('the section keeps its original « Marchés » heading', () => {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    expect(screen.queryByText('Suivis par le moteur')).not.toBeInTheDocument();
  });

  it('lists exactly the registry markets, and no catalogue one', () => {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    for (const spec of MARKET_SPECS) {
      expect(screen.getAllByText(spec.label).length).toBeGreaterThan(0);
    }
    // A market that exists only in the test catalogue must be nowhere to be seen.
    expect(screen.queryByText(/Bitcoin/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Argent \(XAG\/USD\)/)).not.toBeInTheDocument();
    expect(document.querySelectorAll('[data-catalog-only="true"]')).toHaveLength(0);
  });

  it('a search for a catalogue market finds nothing, and says so', () => {
    render(<MarketSelector variant="panel" active={active} onSelect={() => {}} />);
    fireEvent.change(screen.getByLabelText(/Rechercher un marché/i), {
      target: { value: 'BTCUSD' },
    });
    expect(screen.getByText(/Aucun marché ne correspond/)).toBeInTheDocument();
  });
});

describe('MKT-1 — with the flag off, an unknown market takes the ordinary path', () => {
  let fetchSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchSpy = vi.fn(() =>
      Promise.resolve(new Response('{"detail":"Unsupported instrument"}', { status: 400 })),
    );
    vi.stubGlobal('fetch', fetchSpy);
  });
  afterEach(() => vi.unstubAllGlobals());

  it('still requests it and surfaces the backend 400 — no MKT-1 state leaks in', async () => {
    // A stale deep-link to a catalogue market in a normal build must NOT hit the
    // "not followed" copy, which does not exist there: it is an unsupported
    // combination like any other.
    const { result } = renderHook(() => useMarketReading('BTCUSD', 'M15', { source: 'live' }));
    await waitFor(() => expect(result.current.error).toBeTruthy());
    expect(fetchSpy).toHaveBeenCalled();
    expect(result.current.error?.name).toBe('MarketReadingValidationError');
  });
});
