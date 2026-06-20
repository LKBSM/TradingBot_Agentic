import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ComboCard } from '../ComboCard';
import type { ComboMatch } from '@/lib/conditions/types';

function makeMatch(overrides: Partial<ComboMatch> = {}): ComboMatch {
  return {
    instrument: 'XAUUSD',
    timeframe: 'H1',
    candle_close_ts: '2026-05-28T14:00:00+00:00',
    close_price: 2387.4,
    matched: false,
    met_count: 2,
    total: 3,
    conditions_met: [
      { type: 'mtf_aligned', label: '3 TF alignés', met: true, detail: 'Les 3 TF sont alignés.' },
      { type: 'price_in_ob', label: 'Prix dans un Order Block', met: true, detail: 'Prix dans 1 OB.' },
    ],
    conditions_unmet: [
      { type: 'price_in_fvg', label: 'Prix dans un Fair Value Gap', met: false, detail: 'Prix hors FVG.' },
    ],
    context: {
      trend: 'bullish',
      market_phase: 'trend',
      volatility_observed: 'normal',
      mtf_confluence: { h4: 'bullish', h1: 'bullish', m15: 'bullish' },
      bos: { direction: 'bullish', level: 2380, validation_status: 'confirmed' },
      choch: null,
      active_order_blocks: 1,
      active_fair_value_gaps: 0,
      news_upcoming: [],
    },
    ...overrides,
  };
}

describe('ComboCard', () => {
  it('shows both met and unmet conditions (transparent)', () => {
    render(<ComboCard match={makeMatch()} locale="fr" />);
    // met
    expect(screen.getByText('3 TF alignés')).toBeInTheDocument();
    expect(screen.getByText('Prix dans un Order Block')).toBeInTheDocument();
    // unmet
    expect(screen.getByText('Prix dans un Fair Value Gap')).toBeInTheDocument();
    // transparency count
    expect(screen.getByText(/2 de tes 3 conditions/)).toBeInTheDocument();
  });

  it('"Analyser" links to the same market/timeframe in /app', () => {
    render(<ComboCard match={makeMatch({ instrument: 'EURUSD', timeframe: 'H4' })} locale="fr" />);
    const link = screen.getByRole('link', { name: /Analyser/ });
    expect(link).toHaveAttribute('href', '/app?instrument=EURUSD&timeframe=H4');
  });

  it('never uses prescriptive vocabulary ("Trader")', () => {
    const { container } = render(<ComboCard match={makeMatch()} locale="fr" />);
    expect(container.textContent?.toLowerCase()).not.toContain('trader');
  });
});
