import { render as rtlRender, renderHook, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import messages from '@/messages/fr.json';

/**
 * MKT-1 — THE non-negotiable rule: a market with no real data renders an honest
 * empty state and nothing else. No substitute chart, no structure, no OB/FVG, no
 * price, no narrative.
 *
 * The strongest form of that guarantee is not "we don't render the response" but
 * "there is no response": the hooks refuse a catalogue market BEFORE fetching.
 * These tests assert both halves — no request leaves, and the placeholder that
 * shows says the market is not followed.
 */
// The flag is read at module load, so it is stubbed BEFORE the dynamic imports
// below. Nothing is mocked: this exercises the real gate and the real catalogue.
vi.stubEnv('NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST', '1');

const { useMarketReading, useCandles, useLatestPrice, useMtfTrends } = await import(
  '@/lib/market-reading/hooks'
);
const { MarketNotCoveredError } = await import('@/lib/market-reading/api-client');
const { ReadingErrorState, ChartUnavailable } = await import(
  '@/components/app/ReadingPlaceholders'
);
const { resolveComboFromQuery } = await import('@/lib/conditions/app-link');

function render(ui: React.ReactElement) {
  return rtlRender(
    <NextIntlClientProvider locale="fr" messages={messages}>
      {ui}
    </NextIntlClientProvider>,
  );
}

let fetchSpy: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchSpy = vi.fn(() =>
    Promise.reject(new Error('no request should ever be made for a catalogue market')),
  );
  vi.stubGlobal('fetch', fetchSpy);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('MKT-1 — a catalogue market triggers NO request at all', () => {
  it('useMarketReading refuses before fetching, and holds no data', async () => {
    const { result } = renderHook(() => useMarketReading('BTCUSD', 'M15', { source: 'live' }));
    await waitFor(() => expect(result.current.error).toBeInstanceOf(MarketNotCoveredError));
    expect(result.current.data).toBeNull();
    expect(result.current.isLoading).toBe(false);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('useCandles draws no series and asks for none', async () => {
    const { result } = renderHook(() => useCandles('BTCUSD', 'M15', { source: 'live' }));
    await waitFor(() => expect(result.current.error).toBeInstanceOf(MarketNotCoveredError));
    expect(result.current.candles).toBeNull();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('useLatestPrice invents no price', async () => {
    const { result } = renderHook(() => useLatestPrice('BTCUSD', { source: 'live' }));
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.change).toBeNull();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('useMtfTrends reports no alignment rather than a fabricated one', async () => {
    const { result } = renderHook(() => useMtfTrends('BTCUSD', 'M15', { source: 'live' }));
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.trends).toEqual({});
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('a REAL market is still fetched normally — the guard is narrow', async () => {
    renderHook(() => useMarketReading('XAUUSD', 'M15', { source: 'live' }));
    await waitFor(() => expect(fetchSpy).toHaveBeenCalled());
  });
});

describe('MKT-1 — a catalogue market is a REACHABLE view', () => {
  // Regression guard for a real defect the Playwright run caught and the unit
  // tests had missed: resolveComboFromQuery rejected any market outside the real
  // perimeter, so clicking one in the column silently fell back to XAUUSD and the
  // honest empty state was never reachable in the product. Selecting a market and
  // refusing its DATA are two different things.
  it('resolves as a valid combo so the view can say the market is not followed', () => {
    expect(resolveComboFromQuery('BTCUSD', 'M15')).toEqual({
      instrument: 'BTCUSD',
      timeframe: 'M15',
    });
  });

  it('still rejects a genuinely unknown market, and an unknown timeframe', () => {
    expect(resolveComboFromQuery('NOTAMARKET', 'M15')).toBeNull();
    expect(resolveComboFromQuery('BTCUSD', 'M7')).toBeNull();
  });
});

describe('MKT-1 — the empty state names the real reason', () => {
  it('says the market is not followed yet, and names it', () => {
    render(<ReadingErrorState error={new MarketNotCoveredError('BTCUSD')} onRetry={() => {}} />);
    expect(screen.getByText('Pas encore disponible sur ce marché')).toBeInTheDocument();
    // The market is named by its human label, not a raw ticker.
    expect(screen.getByText(/Bitcoin \(BTC\/USD\)/)).toBeInTheDocument();
    expect(screen.getByText(/rien n’est simulé en attendant/)).toBeInTheDocument();
  });

  it('offers NO retry — retrying could not change coverage', () => {
    render(<ReadingErrorState error={new MarketNotCoveredError('BTCUSD')} onRetry={() => {}} />);
    expect(screen.queryByRole('button', { name: /Réessayer/i })).not.toBeInTheDocument();
  });

  it('is NOT the "combinaison non prise en charge" copy — a different fact', () => {
    render(<ReadingErrorState error={new MarketNotCoveredError('BTCUSD')} onRetry={() => {}} />);
    expect(screen.queryByText(/n’est pas prise en charge/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Données indisponibles/)).not.toBeInTheDocument();
  });

  it('renders no number at all — no price, no level, no score', () => {
    const { container } = render(
      <ReadingErrorState error={new MarketNotCoveredError('BTCUSD')} onRetry={() => {}} />,
    );
    // The only digits allowed are the ones inside the market's own name.
    const text = (container.textContent ?? '').replace(/Bitcoin \(BTC\/USD\)/g, '');
    expect(text).not.toMatch(/\d/);
  });

  it('the chart placeholder says the same thing, with no retry', () => {
    render(<ChartUnavailable notCoveredMarket="BTCUSD" onRetry={() => {}} />);
    expect(screen.getByText('Pas encore disponible sur ce marché')).toBeInTheDocument();
    expect(screen.getByText(/Rien n’est dessiné à la place/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Réessayer/i })).not.toBeInTheDocument();
  });

  it('an ordinary chart failure keeps its own copy and its retry', () => {
    render(<ChartUnavailable reason="timeout" onRetry={() => {}} />);
    expect(screen.getByText('Graphique indisponible')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Réessayer/i })).toBeInTheDocument();
  });
});
