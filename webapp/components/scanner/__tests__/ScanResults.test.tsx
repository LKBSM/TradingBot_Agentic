import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ScanResults } from '../ScanResults';
import type { ConditionsConfig, ConditionsScanResponse } from '@/lib/conditions/types';

const CONFIG: ConditionsConfig = {
  logic: 'AND',
  conditions: [{ type: 'trend_is', trend: 'bullish' }],
};

function makeResponse(overrides: Partial<ConditionsScanResponse> = {}): ConditionsScanResponse {
  return {
    as_of: new Date(Date.now() - (3 * 60_000 + 5_000)).toISOString(), // 3 min ago
    logic: 'AND',
    scanned: 1,
    matches: [
      {
        instrument: 'XAUUSD',
        timeframe: 'M15',
        candle_close_ts: new Date(Date.now() - (3 * 60_000 + 5_000)).toISOString(),
        close_price: 2400,
        matched: true,
        met_count: 1,
        total: 1,
        conditions_met: [
          { type: 'trend_is', label: 'Tendance haussière', met: true, detail: 'Tendance haussière.' },
        ],
        conditions_unmet: [],
        context: {
          trend: 'bullish',
          market_phase: 'trend',
          volatility_observed: 'normal',
          mtf_confluence: { h4: 'bullish', h1: 'bullish', m15: 'bullish' },
          bos: null,
          choch: null,
          active_order_blocks: 0,
          active_fair_value_gaps: 0,
          news_upcoming: [],
        },
      },
    ],
    unavailable: [],
    ...overrides,
  };
}

function renderResults(props: Partial<React.ComponentProps<typeof ScanResults>> = {}) {
  const onRefresh = vi.fn();
  const onToggleAutoRefresh = vi.fn();
  const onEdit = vi.fn();
  render(
    <ScanResults
      response={makeResponse()}
      config={CONFIG}
      locale="fr"
      onEdit={onEdit}
      onRefresh={onRefresh}
      isRefreshing={false}
      autoRefreshEnabled
      onToggleAutoRefresh={onToggleAutoRefresh}
      {...props}
    />,
  );
  return { onRefresh, onToggleAutoRefresh, onEdit };
}

describe('ScanResults — freshness & refresh', () => {
  it('shows an honest "Dernière analyse" freshness indicator reflecting real age', () => {
    renderResults();
    const fresh = screen.getByTestId('scan-freshness');
    expect(fresh.textContent).toMatch(/Dernière analyse : il y a 3 min/);
  });

  it('shows "Analyse en cours…" while a scan is running', () => {
    renderResults({ isRefreshing: true });
    expect(screen.getByTestId('scan-freshness').textContent).toMatch(/Analyse en cours/);
  });

  it('"Relancer le scan" forces a refresh', () => {
    const { onRefresh } = renderResults();
    fireEvent.click(screen.getByRole('button', { name: /Relancer le scan/ }));
    expect(onRefresh).toHaveBeenCalledTimes(1);
  });

  it('disables "Relancer" while refreshing', () => {
    renderResults({ isRefreshing: true });
    expect(screen.getByRole('button', { name: /Scan…/ })).toBeDisabled();
  });

  it('exposes the auto-refresh switch reflecting the current preference', () => {
    renderResults({ autoRefreshEnabled: true });
    const sw = screen.getByRole('switch');
    expect(sw).toHaveAttribute('aria-checked', 'true');
  });

  it('toggling the switch flips the preference', () => {
    const { onToggleAutoRefresh } = renderResults({ autoRefreshEnabled: true });
    fireEvent.click(screen.getByRole('switch'));
    expect(onToggleAutoRefresh).toHaveBeenCalledWith(false);
  });
});

/** A full match with an explicit freshness. */
function matchWith(
  instrument: string,
  timeframe: string,
  freshness: 'fresh' | 'aging' | 'stale',
): ConditionsScanResponse['matches'][number] {
  return {
    instrument,
    timeframe,
    candle_close_ts: new Date(Date.now() - 3 * 60_000).toISOString(),
    close_price: 2400,
    matched: true,
    met_count: 1,
    total: 1,
    freshness,
    bars_behind: freshness === 'stale' ? 8 : 0,
    conditions_met: [
      { type: 'trend_is', label: 'Tendance haussière', met: true, detail: 'Tendance haussière.' },
    ],
    conditions_unmet: [],
    context: {
      trend: 'bullish',
      market_phase: 'trend',
      volatility_observed: 'normal',
      mtf_confluence: { h4: 'bullish', h1: 'bullish', m15: 'bullish' },
      bos: null,
      choch: null,
      active_order_blocks: 0,
      active_fair_value_gaps: 0,
      news_upcoming: [],
    },
  };
}

describe('ScanResults — freshness bucketing', () => {
  it('keeps a fresh full match under "présentes maintenant", no older-reading section', () => {
    renderResults({
      response: makeResponse({ matches: [matchWith('XAUUSD', 'M15', 'fresh')] }),
    });
    expect(screen.getByText('Conditions présentes maintenant')).toBeInTheDocument();
    expect(screen.queryByText(/lecture plus ancienne/)).not.toBeInTheDocument();
  });

  it('holds a STALE full match in its own "lecture plus ancienne" section', () => {
    renderResults({
      response: makeResponse({ matches: [matchWith('XAUUSD', 'M15', 'stale')] }),
    });
    expect(screen.getByText(/Sur une lecture plus ancienne — à rafraîchir/)).toBeInTheDocument();
    // The combo DOES meet the conditions (just on an aged reading), so the
    // "aucun marché ne réunit" empty note must NOT appear.
    expect(screen.queryByText(/Aucun marché ne réunit/)).not.toBeInTheDocument();
  });

  it('separates a fresh and a stale full match into the two sections', () => {
    renderResults({
      response: makeResponse({
        scanned: 2,
        matches: [matchWith('XAUUSD', 'M15', 'fresh'), matchWith('EURUSD', 'H4', 'stale')],
      }),
    });
    expect(screen.getByText('Conditions présentes maintenant')).toBeInTheDocument();
    expect(screen.getByText(/Sur une lecture plus ancienne — à rafraîchir/)).toBeInTheDocument();
  });
});
