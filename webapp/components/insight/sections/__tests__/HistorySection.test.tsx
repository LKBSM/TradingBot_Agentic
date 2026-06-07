import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { HistorySection } from '../HistorySection';
import { Accordion } from '@/components/ui/accordion';
import type { HistoricalStats, InsightSignalV2 } from '@/types/insight';

// HistorySection only reads signal.historical_stats — a partial cast is enough.
function renderHistory(h: HistoricalStats | null) {
  return render(
    <Accordion type="multiple" defaultValue={['history']}>
      <HistorySection
        signal={{ historical_stats: h } as unknown as InsightSignalV2}
      />
    </Accordion>,
  );
}

const NULLED_STATS = {
  similar_setups_n: null,
  hit_rate_observed: null,
  profit_factor: null,
  profit_factor_ci95: null,
  empirical_coverage: 0.91,
  backtest_window: '2019–2025',
} as unknown as HistoricalStats;

const REAL_STATS: HistoricalStats = {
  similar_setups_n: 12,
  hit_rate_observed: 0.58,
  profit_factor: 1.3,
  profit_factor_ci95: [1.05, 1.62],
  empirical_coverage: 0.9,
  backtest_window: '2019–2025',
};

describe('HistorySection — null-stat resilience (post-pivot data)', () => {
  it('shows the "no history" message when stat values are null', () => {
    renderHistory(NULLED_STATS);
    expect(
      screen.getByText(/Aucun historique disponible/),
    ).toBeInTheDocument();
    // Must NOT crash on null.toFixed and must not show a "cas" badge.
    expect(screen.queryByText(/cas$/)).not.toBeInTheDocument();
  });

  it('shows the "no history" message when historical_stats is entirely null', () => {
    renderHistory(null);
    expect(
      screen.getByText(/Aucun historique disponible/),
    ).toBeInTheDocument();
  });

  it('renders the figures when real stats are present', () => {
    renderHistory(REAL_STATS);
    expect(screen.getByText('12 cas')).toBeInTheDocument();
    // 1.3 → "1,30" (FR formatting).
    expect(screen.getByText(/1,30/)).toBeInTheDocument();
  });
});
