import { render, screen } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import { PriceFreshnessBadge } from '../PriceFreshnessBadge';
import { formatLocalHm } from '@/lib/time/localTime';

/**
 * diag/price-freshness-zone-card — the reference price a card shows is the last
 * CLOSED candle; without a timestamp a legitimate few-seconds lag is
 * indistinguishable from a real staleness problem. The badge surfaces the epoch
 * (`priceTs`) that already exists internally as an exact local time + age.
 */
describe('PriceFreshnessBadge', () => {
  it('renders nothing when no timestamp is known (honest: no freshness claimed)', () => {
    const { container } = render(<PriceFreshnessBadge tsSec={null} />);
    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByTestId('price-freshness')).toBeNull();
  });

  it('renders nothing for a non-finite timestamp', () => {
    const { container } = render(<PriceFreshnessBadge tsSec={Number.NaN} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows the exact local time the reference price was read', () => {
    const tsSec = Math.floor(Date.now() / 1000) - 600; // 10 min ago
    render(<PriceFreshnessBadge tsSec={tsSec} />);
    const badge = screen.getByTestId('price-freshness');
    // Exact clock time (same formatter as the component → timezone-agnostic here).
    expect(badge).toHaveTextContent(formatLocalHm(new Date(tsSec * 1000)));
    expect(badge).toHaveTextContent(/Prix à/); // fr label from reading.temporal.priceAt
  });

  it('surfaces the relative age after mount so staleness is legible', () => {
    const tsSec = Math.floor(Date.now() / 1000) - 600; // 10 min ago
    render(<PriceFreshnessBadge tsSec={tsSec} />);
    const badge = screen.getByTestId('price-freshness');
    // relativePast → « il y a 10 minutes » (or « à l'instant » at the boundary).
    expect(badge.textContent).toMatch(/il y a|à l'instant/);
  });

  it('carries an explanatory tooltip (gap = elapsed time, not an error)', () => {
    render(<PriceFreshnessBadge tsSec={Math.floor(Date.now() / 1000)} />);
    const badge = screen.getByTestId('price-freshness');
    expect(badge.getAttribute('title')).toMatch(/temps écoulé/);
  });
});
