import { fireEvent, render, screen, waitFor } from '@/components/test-utils';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AppWorkspace } from '../AppWorkspace';
import { ChatProvider } from '@/components/chat/ChatProvider';
import { ChartViewProvider } from '@/lib/chart/viewState';
import { FIXTURE_XAU_M15 } from '@/lib/market-reading/fixtures';
import { __resetReadingRetention } from '@/lib/market-reading/hooks';
import type { Combo } from '@/lib/market-reading/store';

const XAU_M15: Combo = { instrument: 'XAUUSD', timeframe: 'M15' };

// Override only fetchMarketReading; keep the real error classes (ReadingPlaceholders
// uses them for instanceof checks).
const fetchMock = vi.fn();
vi.mock('@/lib/market-reading/api-client', async (importActual) => {
  const actual =
    await importActual<typeof import('@/lib/market-reading/api-client')>();
  return {
    ...actual,
    fetchMarketReading: (...args: unknown[]) => fetchMock(...args),
    // Chart feed is stubbed (chart itself is stubbed below); these tests target
    // the reading/selection flow, not the candle feed.
    fetchCandles: () => Promise.resolve([]),
  };
});

import {
  MarketReadingNotAvailableError,
  MarketReadingValidationError,
} from '@/lib/market-reading/api-client';

// Stub the candlestick chart — lightweight-charts needs a real canvas/layout,
// unavailable in jsdom. These tests exercise the data/selection flow, not the
// chart rendering (covered separately).
vi.mock('@/components/app/ReadingChart', () => ({
  ReadingChart: () => <div data-testid="reading-chart" />,
}));

// AppWorkspace now syncs the active combo with the URL (NAV-01/02). Stub the
// app-router hooks: empty search params → the combo comes from select()/prop as
// before, and router.replace is a no-op in jsdom.
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => '/app',
  useSearchParams: () => new URLSearchParams(),
}));

// These tests target the live (backend) path; force it explicitly since the
// default source is now the local mocks. The desktop selector (rail) and chat
// live in the product shell now, so the combo is seeded via `initialCombo`
// (equivalent to the URL, which /app treats as the source of truth) rather than
// by clicking a nav that the workspace no longer renders on desktop.
function renderApp(initialCombo: Combo | null = null) {
  return render(
    <ChatProvider>
      <ChartViewProvider>
        <AppWorkspace dataSource="live" initialCombo={initialCombo} />
      </ChartViewProvider>
    </ChatProvider>,
  );
}

beforeEach(() => {
  fetchMock.mockReset();
  // DETTE-1: reset PERF-1's module-level SWR retention cache between tests, so a
  // prior test's cached XAU/M15 reading doesn't suppress the first-load skeleton
  // (the caches are module state, shared across tests in this file). This is the
  // harness fix PERF-1 shipped __resetReadingRetention() for; it was wired into
  // hooks.test.ts / useCandles.test.ts but not here.
  __resetReadingRetention();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('AppWorkspace — /app view', () => {
  it('renders the empty state and fires no fetch before any selection', () => {
    renderApp();
    // Centre: empty state. The instrument selector + chat are owned by the
    // product shell now, so the workspace itself renders only the reading area.
    expect(
      screen.getByText(/Sélectionnez une combinaison à gauche/),
    ).toBeInTheDocument();
    // No fetch fired until a combo is selected.
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('fetches and renders the reading for the active combo', async () => {
    fetchMock.mockResolvedValue(FIXTURE_XAU_M15);
    renderApp(XAU_M15);

    await waitFor(() =>
      expect(screen.getByText('Tendance haussière')).toBeInTheDocument(),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      'XAUUSD',
      'M15',
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it('shows a skeleton while the initial fetch is in flight', async () => {
    fetchMock.mockReturnValue(new Promise(() => {})); // never resolves
    renderApp(XAU_M15);

    await waitFor(() =>
      expect(screen.getByTestId('reading-skeleton')).toBeInTheDocument(),
    );
  });

  it('renders a friendly 503 error state with a retry button', async () => {
    fetchMock.mockRejectedValue(
      new MarketReadingNotAvailableError('service not configured'),
    );
    renderApp(XAU_M15);

    await waitFor(() =>
      expect(
        screen.getByText(/n'est pas disponible sur cet environnement/i),
      ).toBeInTheDocument(),
    );
    expect(screen.getByRole('button', { name: /réessayer/i })).toBeInTheDocument();
  });

  it('renders a 400 validation error WITHOUT a retry button', async () => {
    fetchMock.mockRejectedValue(
      new MarketReadingValidationError('unsupported'),
    );
    renderApp(XAU_M15);

    await waitFor(() =>
      expect(
        screen.getByText(/n’est pas prise en charge/),
      ).toBeInTheDocument(),
    );
    expect(
      screen.queryByRole('button', { name: /réessayer/i }),
    ).not.toBeInTheDocument();
  });

  it('retries after an error and renders the reading on success', async () => {
    // First call fails; once the service is back, every subsequent call resolves
    // — including the multi-timeframe reads the always-visible desktop regime
    // panel fires (useMtfTrends), so use a persistent resolve, not `…Once`.
    fetchMock
      .mockRejectedValueOnce(new MarketReadingNotAvailableError('down'))
      .mockResolvedValue(FIXTURE_XAU_M15);
    renderApp(XAU_M15);

    const retry = await screen.findByRole('button', { name: /réessayer/i });
    fireEvent.click(retry);

    await waitFor(() =>
      expect(screen.getByText('Tendance haussière')).toBeInTheDocument(),
    );
  });
});
