import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ReadingColumn } from '../ReadingColumn';
import { MarketReadingNotAvailableError } from '@/lib/market-reading/api-client';
import {
  FIXTURE_QUIET_XAU_H4,
  FIXTURE_XAU_M15,
} from '@/lib/market-reading/fixtures';
import type { Combo } from '@/lib/market-reading/store';

// Stub the canvas chart so we can assert it is mounted without a real renderer.
vi.mock('@/components/app/ReadingChart', () => ({
  ReadingChart: () => <div data-testid="reading-chart" />,
}));

const XAU_M15: Combo = { instrument: 'XAUUSD', timeframe: 'M15' };
const XAU_H4: Combo = { instrument: 'XAUUSD', timeframe: 'H4' };

const noop = () => {};

describe('ReadingColumn', () => {
  it('shows the empty state when no combo is selected', () => {
    render(
      <ReadingColumn
        active={null}
        reading={null}
        isLoading={false}
        isRefreshing={false}
        error={null}
        onRetry={noop}
      />,
    );
    expect(
      screen.getByText(/Sélectionnez une combinaison/),
    ).toBeInTheDocument();
  });

  it('renders the "Données indisponibles" error card with a retry', () => {
    render(
      <ReadingColumn
        active={XAU_M15}
        reading={null}
        isLoading={false}
        isRefreshing={false}
        error={new MarketReadingNotAvailableError('down')}
        onRetry={noop}
      />,
    );
    expect(screen.getByText('Données indisponibles')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /réessayer/i }),
    ).toBeInTheDocument();
  });

  it('mounts the chart when a candle feed is available (XAU M15)', async () => {
    render(
      <ReadingColumn
        active={XAU_M15}
        reading={FIXTURE_XAU_M15}
        isLoading={false}
        isRefreshing={false}
        error={null}
        onRetry={noop}
      />,
    );
    await waitFor(() =>
      expect(screen.getByTestId('reading-chart')).toBeInTheDocument(),
    );
  });

  it('shows the "Graphique indisponible" placeholder when the feed is missing (XAU H4)', () => {
    render(
      <ReadingColumn
        active={XAU_H4}
        reading={FIXTURE_QUIET_XAU_H4}
        isLoading={false}
        isRefreshing={false}
        error={null}
        onRetry={noop}
      />,
    );
    expect(screen.getByText('Graphique indisponible')).toBeInTheDocument();
    expect(screen.queryByTestId('reading-chart')).not.toBeInTheDocument();
  });
});
