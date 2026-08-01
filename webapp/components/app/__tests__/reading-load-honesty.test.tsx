import { describe, expect, it, vi, afterEach } from 'vitest';
import { render, screen, act } from '@/components/test-utils';
import { ReadingErrorState, SlowLoadHint } from '../ReadingPlaceholders';
import {
  MarketReadingError,
  MarketReadingNoDataError,
  MarketReadingNotAvailableError,
  MarketReadingValidationError,
} from '@/lib/market-reading/api-client';

/**
 * PERF-1 honesty-of-loading: the reading error card must say WHICH failure it is
 * — trop lent / serveur injoignable / aucune donnée / combo non supporté / erreur
 * interne — never one vague "données indisponibles" for all. And a slow load must
 * announce itself so it never reads as a frozen screen.
 */

afterEach(() => {
  vi.useRealTimers();
});

describe('ReadingErrorState — distinct copy per failure mode', () => {
  const noop = () => {};

  it('timeout → "trop de temps à répondre"', () => {
    render(<ReadingErrorState error={new MarketReadingError(0, 'x', 'timeout')} onRetry={noop} />);
    expect(screen.getByText(/trop de temps à répondre/i)).toBeInTheDocument();
    expect(screen.getByText(/Réessayer/i)).toBeInTheDocument();
  });

  it('network → "serveur ne répond pas"', () => {
    render(<ReadingErrorState error={new MarketReadingError(0, 'x', 'network')} onRetry={noop} />);
    expect(screen.getByText(/ne répond pas/i)).toBeInTheDocument();
  });

  it('404 → "aucune donnée ... pour cette combinaison"', () => {
    render(<ReadingErrorState error={new MarketReadingNoDataError('x')} onRetry={noop} />);
    expect(screen.getByText(/aucune donnée.*cette combinaison/i)).toBeInTheDocument();
  });

  it('503 → "service ... pas disponible sur cet environnement"', () => {
    render(<ReadingErrorState error={new MarketReadingNotAvailableError('x')} onRetry={noop} />);
    expect(screen.getByText(/pas disponible sur cet environnement/i)).toBeInTheDocument();
  });

  it('400 (unsupported combo) → validation copy AND no retry button', () => {
    render(<ReadingErrorState error={new MarketReadingValidationError('x')} onRetry={noop} />);
    expect(screen.getByText(/n.?est pas prise en charge/i)).toBeInTheDocument();
    expect(screen.queryByText(/Réessayer/i)).not.toBeInTheDocument();
  });

  it('generic server error (5xx) → generic copy, distinct from the above', () => {
    render(<ReadingErrorState error={new MarketReadingError(500, 'x', 'server')} onRetry={noop} />);
    expect(screen.getByText(/n.?a pas pu être récupérée/i)).toBeInTheDocument();
  });
});

describe('SlowLoadHint — progressive loading honesty', () => {
  it('renders nothing before the threshold, then the "prend plus de temps" note', () => {
    vi.useFakeTimers();
    render(<SlowLoadHint afterMs={6000} />);
    // Before the threshold: silent (the caller shows the skeleton).
    expect(screen.queryByText(/prend plus de temps/i)).not.toBeInTheDocument();
    act(() => {
      vi.advanceTimersByTime(6000);
    });
    expect(screen.getByText(/prend plus de temps/i)).toBeInTheDocument();
  });
});
