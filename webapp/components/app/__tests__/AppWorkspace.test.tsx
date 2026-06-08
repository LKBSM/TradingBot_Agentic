import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AppWorkspace } from '../AppWorkspace';
import { ChatProvider } from '@/components/chat/ChatProvider';
import { FIXTURE_XAU_M15 } from '@/lib/market-reading/fixtures';

// Override only fetchMarketReading; keep the real error classes (ReadingPlaceholders
// uses them for instanceof checks).
const fetchMock = vi.fn();
vi.mock('@/lib/market-reading/api-client', async (importActual) => {
  const actual =
    await importActual<typeof import('@/lib/market-reading/api-client')>();
  return { ...actual, fetchMarketReading: (...args: unknown[]) => fetchMock(...args) };
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

// These tests target the live (backend) path; force it explicitly since the
// default source is now the local mocks.
function renderApp() {
  return render(
    <ChatProvider>
      <AppWorkspace dataSource="live" />
    </ChatProvider>,
  );
}

beforeEach(() => {
  fetchMock.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('AppWorkspace — /app view', () => {
  it('renders the three columns with the empty state before any selection', () => {
    renderApp();
    // Left: instruments nav with both markets.
    expect(
      screen.getByRole('navigation', { name: /combinaisons disponibles/i }),
    ).toBeInTheDocument();
    expect(screen.getByText('Or (XAU/USD)')).toBeInTheDocument();
    expect(screen.getByText('Euro / Dollar (EUR/USD)')).toBeInTheDocument();
    // Centre: empty state.
    expect(
      screen.getByText(/Sélectionnez une combinaison à gauche/),
    ).toBeInTheDocument();
    // Right: chat sidebar, idle context.
    expect(
      screen.getByRole('complementary', { name: /assistant sentinel/i }),
    ).toBeInTheDocument();
    // No fetch fired until a combo is selected.
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('lists the 6 V1 combos (2 instruments × 3 timeframes)', () => {
    renderApp();
    const nav = screen.getByRole('navigation', {
      name: /combinaisons disponibles/i,
    });
    // Each combo row has a select button + a pin toggle. Count the select
    // buttons only (the pin toggles carry an "Épingler/Désépingler" label).
    const selectButtons = within(nav)
      .getAllByRole('button')
      .filter((b) => !/épingler/i.test(b.getAttribute('aria-label') ?? ''));
    expect(selectButtons).toHaveLength(6);
  });

  it('fetches and renders the reading when a combo is selected', async () => {
    fetchMock.mockResolvedValue(FIXTURE_XAU_M15);
    renderApp();

    // Select XAUUSD · 15 minutes (first timeframe of the first market).
    const nav = screen.getByRole('navigation', {
      name: /combinaisons disponibles/i,
    });
    fireEvent.click(within(nav).getAllByRole('button')[0]!);

    await waitFor(() =>
      expect(screen.getByText('Tendance haussière')).toBeInTheDocument(),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      'XAUUSD',
      'M15',
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    // Chat context now reflects the selected combo.
    const chat = screen.getByRole('complementary', { name: /assistant sentinel/i });
    expect(within(chat).getByText(/Or \(XAU\/USD\) · 15 minutes/)).toBeInTheDocument();
  });

  it('shows a skeleton while the initial fetch is in flight', async () => {
    fetchMock.mockReturnValue(new Promise(() => {})); // never resolves
    renderApp();
    const nav = screen.getByRole('navigation', {
      name: /combinaisons disponibles/i,
    });
    fireEvent.click(within(nav).getAllByRole('button')[0]!);

    await waitFor(() =>
      expect(screen.getByTestId('reading-skeleton')).toBeInTheDocument(),
    );
  });

  it('renders a friendly 503 error state with a retry button', async () => {
    fetchMock.mockRejectedValue(
      new MarketReadingNotAvailableError('service not configured'),
    );
    renderApp();
    const nav = screen.getByRole('navigation', {
      name: /combinaisons disponibles/i,
    });
    fireEvent.click(within(nav).getAllByRole('button')[0]!);

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
    renderApp();
    const nav = screen.getByRole('navigation', {
      name: /combinaisons disponibles/i,
    });
    fireEvent.click(within(nav).getAllByRole('button')[0]!);

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
    fetchMock
      .mockRejectedValueOnce(new MarketReadingNotAvailableError('down'))
      .mockResolvedValueOnce(FIXTURE_XAU_M15);
    renderApp();
    const nav = screen.getByRole('navigation', {
      name: /combinaisons disponibles/i,
    });
    fireEvent.click(within(nav).getAllByRole('button')[0]!);

    const retry = await screen.findByRole('button', { name: /réessayer/i });
    fireEvent.click(retry);

    await waitFor(() =>
      expect(screen.getByText('Tendance haussière')).toBeInTheDocument(),
    );
  });
});
