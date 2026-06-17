import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Accordion } from '@/components/ui/accordion';
import { RegimeSection } from '../RegimeSection';
import type { MarketReadingRegime } from '@/types/market-reading';
import type { MtfTrendMap } from '@/lib/market-reading/mtf-trend';

// Mock the read-only MTF hook so the section renders deterministically without
// any network — the fetch wiring is covered separately; here we assert display.
const mockUseMtfTrends = vi.fn();
vi.mock('@/lib/market-reading/hooks', () => ({
  useMtfTrends: (instrument: string | null) => mockUseMtfTrends(instrument),
}));

const REGIME: MarketReadingRegime = {
  trend: 'bullish',
  volatility_observed: 'normal',
  market_phase: 'trend',
  mtf_confluence: {},
};

function renderSection(trends: MtfTrendMap, isLoading = false) {
  mockUseMtfTrends.mockReturnValue({ trends, isLoading });
  return render(
    <Accordion type="multiple" defaultValue={['regime']}>
      <RegimeSection regime={REGIME} instrument="XAUUSD" />
    </Accordion>,
  );
}

beforeEach(() => mockUseMtfTrends.mockReset());

describe('RegimeSection (MTF trend alignment)', () => {
  it('keeps the panel title and the volatility badge', () => {
    renderSection({ h4: 'bullish', h1: 'bullish', m15: 'bullish' });
    expect(screen.getByText('Régime de marché')).toBeInTheDocument();
    expect(screen.getByText('Volatilité normale')).toBeInTheDocument();
  });

  it('shows the three timeframes at a glance with arrows', () => {
    renderSection({ h4: 'bullish', h1: 'bullish', m15: 'bearish' });
    expect(screen.getByText(/H4\s*↗/)).toBeInTheDocument();
    expect(screen.getByText(/H1\s*↗/)).toBeInTheDocument();
    expect(screen.getByText(/M15\s*↘/)).toBeInTheDocument();
  });

  it('renders the descriptive relation line', () => {
    renderSection({ h4: 'bullish', h1: 'bullish', m15: 'bearish' });
    expect(
      screen.getByText('M15 se replie contre la tendance H4 haussière.'),
    ).toBeInTheDocument();
  });

  it('keeps a descriptive (non-instruction) disclaimer', () => {
    renderSection({ h4: 'neutral', h1: 'neutral', m15: 'neutral' });
    expect(
      screen.getByText(/ne constitue pas une instruction adressée au trader/i),
    ).toBeInTheDocument();
  });

  it('no longer restates the redundant trend / market-phase badges', () => {
    renderSection({ h4: 'bullish', h1: 'bullish', m15: 'bullish' });
    expect(screen.queryByText('Tendance haussière')).not.toBeInTheDocument();
    expect(screen.queryByText('Phase de tendance')).not.toBeInTheDocument();
  });

  it('degrades gracefully when no timeframe is available', () => {
    renderSection({ h4: null, h1: null, m15: null });
    expect(
      screen.getByText('Alignement multi-timeframe indisponible.'),
    ).toBeInTheDocument();
  });
});
