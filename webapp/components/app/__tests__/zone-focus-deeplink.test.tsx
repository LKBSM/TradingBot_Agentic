import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AppWorkspace } from '../AppWorkspace';
import { ChatProvider } from '@/components/chat/ChatProvider';
import { ChartViewProvider, useChartViewOptional } from '@/lib/chart/viewState';
import { FIXTURE_XAU_M15 } from '@/lib/market-reading/fixtures';

const fetchMock = vi.fn();
vi.mock('@/lib/market-reading/api-client', async (importActual) => {
  const actual =
    await importActual<typeof import('@/lib/market-reading/api-client')>();
  return {
    ...actual,
    fetchMarketReading: (...args: unknown[]) => fetchMock(...args),
    fetchCandles: () => Promise.resolve([]),
  };
});

// Chart needs a real canvas; stub it — we assert the focus COMMAND, not pixels.
vi.mock('@/components/app/ReadingChart', () => ({
  ReadingChart: () => <div data-testid="reading-chart" />,
}));

/** Surfaces the chart's one-shot focus command (zone id) from shared state. */
function FocusProbe() {
  const { view } = useChartViewOptional();
  return <div data-testid="focus-zone">{view.focus?.zoneId ?? ''}</div>;
}

function renderWithFocus(focusZoneId: string | null) {
  return render(
    <ChatProvider>
      <ChartViewProvider>
        <AppWorkspace
          initialCombo={{ instrument: 'XAUUSD', timeframe: 'M15' }}
          initialFocusZoneId={focusZoneId}
          dataSource="live"
        />
        <FocusProbe />
      </ChartViewProvider>
    </ChatProvider>,
  );
}

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(FIXTURE_XAU_M15);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('Analyser deep-link → chart focus', () => {
  it('focuses the zone whose real id arrives via ?focus= (no stale-zone notice)', async () => {
    renderWithFocus('ob-xau-1');
    await waitFor(() =>
      expect(screen.getByTestId('focus-zone')).toHaveTextContent('ob-xau-1'),
    );
    // A resolved focus never triggers the "no longer detected" notice.
    expect(screen.queryByText(/n’est plus détectée/)).not.toBeInTheDocument();
  });

  it('ignores an unknown/stale id (id-lock) and says so discreetly — never redraws it', async () => {
    renderWithFocus('not-a-real-zone');
    // Once the reading resolves, the honest notice appears; the app opened
    // normally on the combo and NO focus command was dispatched.
    expect(await screen.findByText(/n’est plus détectée/)).toBeInTheDocument();
    expect(screen.getByTestId('focus-zone')).toHaveTextContent('');
  });

  it('shows no notice at all without a ?focus= param', async () => {
    renderWithFocus(null);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(screen.queryByText(/n’est plus détectée/)).not.toBeInTheDocument();
  });
});
