import { render as rtlRender, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import { ComboCard } from '../ComboCard';
import { ScanResults } from '../ScanResults';
import type { ComboMatch, ConditionsConfig, ConditionsScanResponse } from '@/lib/conditions/types';
import messages from '@/messages/fr.json';

function render(ui: React.ReactElement) {
  return rtlRender(
    <NextIntlClientProvider locale="fr" messages={messages}>
      {ui}
    </NextIntlClientProvider>,
  );
}

function baseContext(): ComboMatch['context'] {
  return {
    trend: 'bullish',
    market_phase: 'trend',
    volatility_observed: 'normal',
    mtf_confluence: {},
    mtf_trends: { h4: 'bullish', h1: 'bullish', m15: 'bullish' },
    bos: null,
    choch: null,
    active_order_blocks: 0,
    active_fair_value_gaps: 0,
    structural_range: null,
    news_upcoming: [],
  };
}

function combo(overrides: Partial<ComboMatch> = {}): ComboMatch {
  return {
    instrument: 'XAUUSD',
    timeframe: 'M15',
    candle_close_ts: new Date().toISOString(),
    close_price: 2400,
    matched: true,
    met_count: 1,
    total: 1,
    non_evaluable_count: 0,
    conditions_met: [{ type: 'trend_is', label: 'La tendance', met: true, detail: 'Tendance haussière.' }],
    conditions_unmet: [],
    conditions_non_evaluable: [],
    context: baseContext(),
    freshness: 'fresh',
    bars_behind: 0,
    ...overrides,
  };
}

describe('ComboCard — three blocks (SC-1)', () => {
  it('always renders the « ce qui va à l\'encontre » block, even with nothing against', () => {
    render(<ComboCard match={combo()} locale="fr" now={Date.now()} />);
    // Non-maskable: present regardless of whether anything goes against.
    expect(screen.getByTestId('against-block')).toBeInTheDocument();
    expect(screen.getByText('Ce qui va à l’encontre')).toBeInTheDocument();
    expect(screen.getByText('Ce qui correspond')).toBeInTheDocument();
    expect(screen.getByText('Contexte que tu n’as pas demandé')).toBeInTheDocument();
  });

  it('renders a non-evaluable condition in its own block, not as met nor against', () => {
    const m = combo({
      met_count: 1,
      total: 1,
      non_evaluable_count: 1,
      conditions_non_evaluable: [
        { type: 'last_event_is', label: 'Dernier événement', met: false, available: false, detail: 'Aucun événement daté.' },
      ],
    });
    render(<ComboCard match={m} locale="fr" now={Date.now()} />);
    expect(screen.getByText(/Non évaluable ici/)).toBeInTheDocument();
    // The denominator shown counts only evaluable conditions (1), plus the
    // explicit non-evaluable tally.
    expect(screen.getByText(/1 non évaluable/)).toBeInTheDocument();
  });

  it('renders enriched against-signals in the against block on a FULL match', () => {
    const m = combo({
      matched: true, met_count: 1, total: 1,
      conditions_unmet: [],
      context_against: [
        { label: 'Le 4 h est en tendance haussière', detail: 'désaccord multi-unités' },
        { label: 'La volatilité est contractée', detail: '7 dernières vs 20' },
      ],
    });
    render(<ComboCard match={m} locale="fr" now={Date.now()} />);
    // The block is present AND carries the against-signals (not the empty note).
    expect(screen.getByTestId('against-block')).toBeInTheDocument();
    expect(screen.getByText('Le 4 h est en tendance haussière')).toBeInTheDocument();
    expect(screen.getByText(/désaccord multi-unités/)).toBeInTheDocument();
    expect(screen.queryByText(/Rien ne va à l’encontre/)).not.toBeInTheDocument();
  });

  it('offers « Analyser » and « Ouvrir dans le graphique », never « Trader »', () => {
    render(<ComboCard match={combo()} locale="fr" now={Date.now()} />);
    expect(screen.getByText(/Analyser/)).toBeInTheDocument();
    expect(screen.getByText('Ouvrir dans le graphique')).toBeInTheDocument();
    expect(screen.queryByText(/Trader/)).not.toBeInTheDocument();
  });
});

describe('ScanResults — no-combo state & filters (SC-1)', () => {
  const CONFIG: ConditionsConfig = {
    logic: 'AND',
    conditions: [{ type: 'trend_is', trend: 'bullish' }, { type: 'price_in_ob' }],
  };

  function noComboResponse(): ConditionsScanResponse {
    return {
      as_of: new Date().toISOString(),
      logic: 'AND',
      scanned: 2,
      matches: [
        combo({
          instrument: 'XAUUSD', timeframe: 'M15', matched: false, met_count: 1, total: 2,
          conditions_met: [{ type: 'trend_is', label: 'La tendance', met: true, detail: 'Haussière.' }],
          conditions_unmet: [{ type: 'price_in_ob', label: 'Prix dans un OB', met: false, detail: 'Hors OB.' }],
        }),
        combo({
          instrument: 'EURUSD', timeframe: 'H4', matched: false, met_count: 0, total: 2,
          conditions_met: [],
          conditions_unmet: [
            { type: 'trend_is', label: 'La tendance', met: false, detail: 'Baissière.' },
            { type: 'price_in_ob', label: 'Prix dans un OB', met: false, detail: 'Hors OB.' },
          ],
        }),
      ],
      unavailable: [],
    };
  }

  function renderResults(response: ConditionsScanResponse) {
    render(
      <ScanResults
        response={response}
        config={CONFIG}
        locale="fr"
        onEdit={() => {}}
        onRefresh={() => {}}
        isRefreshing={false}
        autoRefreshEnabled={false}
        onToggleAutoRefresh={() => {}}
      />,
    );
  }

  it('says it is not an error and lists isolated per-condition counts that do not add up', () => {
    renderResults(noComboResponse());
    expect(screen.getByTestId('scan-no-combo')).toBeInTheDocument();
    expect(screen.getByText(/Ce n’est pas une erreur/)).toBeInTheDocument();
    expect(screen.getByText(/ne s’additionnent pas/)).toBeInTheDocument();
    // trend_is met in isolation on 1 combo (XAU); price_in_ob on 0.
    expect(screen.getByText('1 combo')).toBeInTheDocument();
  });

  it('never proposes to loosen or remove a condition (product line #3)', () => {
    renderResults(noComboResponse());
    // The explicit reassurance is shown (the word « assouplir » appears only
    // inside this denial, never as an action)…
    expect(screen.getByText(/ne propose pas de/)).toBeInTheDocument();
    // …and there is NO actionable affordance to relax / remove / widen a condition.
    expect(screen.queryByRole('button', { name: /assouplir|relâcher|retirer|élargir/i })).toBeNull();
    expect(screen.queryByRole('link', { name: /assouplir|relâcher|retirer|élargir/i })).toBeNull();
  });

  it('renders the four display filters, each with its count, and never a sort control', () => {
    renderResults(noComboResponse());
    // Each label may appear both as a filter chip and as a section header — the
    // point is the count is shown per group and there is no sort control.
    expect(screen.getByText(/Correspondances \(0\)/)).toBeInTheDocument();
    expect(screen.getAllByText(/Presque \(1\)/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/Non correspondants \(1\)/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/Non évaluables \(0\)/)).toBeInTheDocument();
    expect(screen.queryByText(/Trier/)).not.toBeInTheDocument();
  });
});
