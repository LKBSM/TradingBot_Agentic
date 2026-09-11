import { render, screen } from '@/components/test-utils';
import { describe, expect, it, vi } from 'vitest';
import { ChatProvider } from '@/components/chat/ChatProvider';

/**
 * MKT-1b — M.I.A must not offer to analyse a market that has no data.
 *
 * Found by eye on the MKT-1 Playwright capture: while the centre column
 * honestly said « Pas encore disponible sur ce marché », the M.I.A panel still
 * proposed « Décompose la structure actuelle » and « Montre-moi les Order Blocks
 * actifs » for that same market. M.I.A could not have invented anything (the
 * backend does not know the market), but the suggestions invited a question
 * about data that does not exist — the dishonesty the empty state exists to
 * avoid.
 *
 * The flag is stubbed BEFORE the dynamic imports so the real gate is exercised.
 */
vi.stubEnv('NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST', '1');

const { AppChatSidebar } = await import('@/components/app/AppChatSidebar');
const { isCatalogOnly } = await import('@/lib/market-catalog');

function renderSidebar(combo: { instrument: string; timeframe: string } | null) {
  return render(
    <ChatProvider>
      <AppChatSidebar active={combo} />
    </ChatProvider>,
  );
}

describe('MKT-1b — M.I.A on an uncovered market', () => {
  it('the fixture really is a catalogue-only market (guards the premise)', () => {
    expect(isCatalogOnly('BTCUSD')).toBe(true);
    expect(isCatalogOnly('XAUUSD')).toBe(false);
  });

  it('offers NO starter question — nothing to analyse, nothing to propose', () => {
    renderSidebar({ instrument: 'BTCUSD', timeframe: 'M15' });
    expect(screen.queryByText(/Décompose la structure actuelle/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Order Blocks actifs/)).not.toBeInTheDocument();
    expect(screen.queryByText(/C'est quoi un CHOCH/)).not.toBeInTheDocument();
  });

  it('says why, naming the market by its human label', () => {
    renderSidebar({ instrument: 'BTCUSD', timeframe: 'M15' });
    expect(screen.getByText('Aucune lecture sur ce marché')).toBeInTheDocument();
    expect(screen.getByText(/Le moteur ne suit pas encore Bitcoin \(BTC\/USD\)/)).toBeInTheDocument();
  });

  it('names the market in the header by its label, not a raw ticker', () => {
    const { container } = renderSidebar({ instrument: 'BTCUSD', timeframe: 'M15' });
    expect(container.textContent).toContain('Bitcoin (BTC/USD)');
  });

  it('a FOLLOWED market keeps its starter questions — the guard is narrow', () => {
    renderSidebar({ instrument: 'XAUUSD', timeframe: 'M15' });
    expect(screen.getByText(/Décompose la structure actuelle/)).toBeInTheDocument();
    expect(screen.queryByText('Aucune lecture sur ce marché')).not.toBeInTheDocument();
  });

  it('no combo at all still behaves as before (idle welcome, no starters)', () => {
    renderSidebar(null);
    expect(screen.queryByText(/Décompose la structure actuelle/)).not.toBeInTheDocument();
    expect(screen.queryByText('Aucune lecture sur ce marché')).not.toBeInTheDocument();
  });
});
