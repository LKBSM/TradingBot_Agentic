import { render as rtlRender, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import { ComboCard } from '../ComboCard';
import type { ComboMatch } from '@/lib/conditions/types';
import messages from '@/messages/fr.json';

// ComboCard now consumes the `scanner` message namespace; fr messages keep the
// asserted FR labels intact.
function render(ui: React.ReactElement) {
  return rtlRender(
    <NextIntlClientProvider locale="fr" messages={messages}>
      {ui}
    </NextIntlClientProvider>,
  );
}

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

  it('counts ONLY high-impact upcoming news (ignores medium/low noise)', () => {
    const match = makeMatch({
      context: {
        ...makeMatch().context,
        news_upcoming: [
          { event: 'US NFP', impact: 'high', time_to_event_min: 30 },
          { event: 'EU PMI', impact: 'medium', time_to_event_min: 45 },
          { event: 'US Jobless', impact: 'low', time_to_event_min: 60 },
        ],
      },
    });
    render(<ComboCard match={match} locale="fr" />);
    // 1 high-impact → singular "importante", medium/low excluded from the count.
    expect(
      screen.getByText(/1 actu importante à venir — à garder en tête/),
    ).toBeInTheDocument();
  });

  it('pluralises the label for several high-impact news', () => {
    const match = makeMatch({
      context: {
        ...makeMatch().context,
        news_upcoming: [
          { event: 'US NFP', impact: 'high', time_to_event_min: 30 },
          { event: 'US CPI', impact: 'high', time_to_event_min: 90 },
        ],
      },
    });
    render(<ComboCard match={match} locale="fr" />);
    expect(
      screen.getByText(/2 actus importantes à venir — à garder en tête/),
    ).toBeInTheDocument();
  });

  it('hides the heads-up line when no high-impact news is upcoming', () => {
    const match = makeMatch({
      context: {
        ...makeMatch().context,
        news_upcoming: [
          { event: 'EU PMI', impact: 'medium', time_to_event_min: 45 },
          { event: 'US Jobless', impact: 'low', time_to_event_min: 60 },
        ],
      },
    });
    render(<ComboCard match={match} locale="fr" />);
    expect(screen.queryByText(/à garder en tête/)).not.toBeInTheDocument();
  });
});
