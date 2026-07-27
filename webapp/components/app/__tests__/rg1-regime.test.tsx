import { render, screen, fireEvent, waitFor, within } from '@/components/test-utils';
import { afterEach, describe, expect, it, vi } from 'vitest';
import * as React from 'react';
import { RegimeCard } from '../RegimeCard';
import { ChartViewProvider, useChartViewOptional } from '@/lib/chart/viewState';
import { coerceViewAction } from '@/lib/chart/viewActions';
import type { Candle, MarketReadingStructure, MarketStatusPayload } from '@/types/market-reading';

// Deterministic sibling-TF trends (no network).
vi.mock('@/lib/market-reading/hooks', async (orig) => {
  const actual = await orig<typeof import('@/lib/market-reading/hooks')>();
  return {
    ...actual,
    useMtfTrends: () => ({
      trends: { h4: 'bearish', h1: 'bearish', m15: 'bullish' },
      isLoading: false,
    }),
  };
});

// D1/W1 reference candles (ascending). 3 each → all 6 reference levels resolve.
const DAILY: Candle[] = [
  { time: 1, open: 2410, high: 2424.8, low: 2370.5, close: 2415 },
  { time: 2, open: 2415, high: 2421.2, low: 2376.2, close: 2400 },
  { time: 3, open: 2398.6, high: 2405, low: 2390, close: 2392 },
];
const WEEKLY: Candle[] = [
  { time: 10, open: 2380, high: 2450, low: 2350, close: 2420 },
  { time: 20, open: 2420, high: 2424.8, low: 2370.5, close: 2405 },
  { time: 30, open: 2411.4, high: 2412, low: 2390, close: 2392 },
];
const fetchCandlesMock = vi.fn((_i: string, tf: string) =>
  Promise.resolve(tf === 'W1' ? WEEKLY : DAILY),
);
vi.mock('@/lib/market-reading/api-client', async (orig) => {
  const actual = await orig<typeof import('@/lib/market-reading/api-client')>();
  return { ...actual, fetchCandles: (i: string, tf: string) => fetchCandlesMock(i, tf) };
});

afterEach(() => vi.clearAllMocks());

const FULL_STRUCT = {
  order_blocks: [
    { id: 'ob1', direction: 'bearish', level_high: 2401.1, level_low: 2398.4, importance: 'high', status: 'active', created_at: '2026-07-26T11:00:00Z', tested: true, user_flagged: false },
    { id: 'ob2', direction: 'bullish', level_high: 2385.5, level_low: 2383.0, importance: 'medium', status: 'mitigated', created_at: '2026-07-26T09:00:00Z', tested: true, mitigated_at: '2026-07-26T12:45:00Z', user_flagged: false },
  ],
  fair_value_gaps: [
    { id: 'fvg1', direction: 'bullish', level_high: 2394.0, level_low: 2389.0, status: 'partially_filled', created_at: '2026-07-26T10:15:00Z', tested: true, fill_level: 2391.9, user_flagged: false },
  ],
  liquidity_pools: [
    { id: 'lqh', side: 'bsl', kind: 'range_high', level: 2421.2, touches: 1, is_external: true, status: 'intact', created_at: '2026-07-26T08:00:00Z', user_flagged: false },
    { id: 'lql', side: 'ssl', kind: 'range_low', level: 2370.5, touches: 1, is_external: true, status: 'intact', created_at: '2026-07-26T08:00:00Z', user_flagged: false },
  ],
  choch_events: [{ level: 2400.8, direction: 'bearish', broken_at: '2026-07-26T14:00:00Z', validation_status: 'confirmed' }],
  bos_events: [{ level: 2399.0, direction: 'bearish', broken_at: '2026-07-26T15:00:00Z', validation_status: 'confirmed' }],
} as unknown as MarketReadingStructure;

const MINIMAL_STRUCT = {
  order_blocks: [],
  fair_value_gaps: [],
  liquidity_pools: [],
  choch_events: [],
  bos_events: [],
} as unknown as MarketReadingStructure;

const HEADER = {
  instrument: 'XAUUSD',
  timeframe: 'H1',
  close_price: 2392.35,
  candle_close_ts: '2026-07-26T21:00:00Z',
} as never;
const REGIME = { trend: 'bearish', volatility_observed: 'normal', market_phase: 'trend', mtf_confluence: {}, volatility_detail: { recent_avg: 3.42, baseline_avg: 3.6, ratio: 0.95, recent_n: 7, baseline_n: 20, threshold_low: 0.7, threshold_high: 1.3 } } as never;
const OPEN_STATUS: MarketStatusPayload = {
  state: 'open',
  reason: 'open',
  instrument: 'XAUUSD',
  timeframe: 'H1',
  last_close_ts: null,
  next_open_ts: null,
  bars_behind: 0,
  continuous: false,
  session_tz: 'America/New_York',
  sessions: [
    { name: 'asia', start: '19:00', end: '04:00' },
    { name: 'london', start: '03:00', end: '11:30' },
    { name: 'new_york', start: '08:00', end: '17:00' },
  ],
  weekly_close: { weekday: 4, time: '17:00' },
} as never;

function Harness({
  structure,
  marketStatus,
}: {
  structure: MarketReadingStructure;
  marketStatus: MarketStatusPayload | null;
}) {
  const [openHelp, setOpenHelp] = React.useState<string | null>(null);
  const onToggleHelp = (k: string) => setOpenHelp((cur) => (cur === k ? null : k));
  return (
    <ChartViewProvider>
      <RefProbe />
      <div className="app-shell">
        <RegimeCard
          regime={REGIME}
          structure={structure}
          header={HEADER}
          price={2392.35}
          marketStatus={marketStatus}
          openHelp={openHelp}
          onToggleHelp={onToggleHelp}
        />
      </div>
    </ChartViewProvider>
  );
}

function RefProbe() {
  const { referenceLevel } = useChartViewOptional();
  return (
    <div data-testid="ref">
      {referenceLevel ? `${referenceLevel.label}=${referenceLevel.price}` : 'none'}
    </div>
  );
}

describe('RG-1 — RegimeCard tiles', () => {
  it('drops a measure with no engine datum (no tile, no « N/A », no 0)', () => {
    render(<Harness structure={MINIMAL_STRUCT} marketStatus={null} />);
    // Present (always have data):
    expect(screen.getByText('Phase de marché')).toBeInTheDocument();
    expect(screen.getByText('Densité')).toBeInTheDocument();
    // Absent (no datum): maturity, last event, position, session, levels.
    expect(screen.queryByText('Maturité')).toBeNull();
    expect(screen.queryByText('Dernier événement')).toBeNull();
    expect(screen.queryByText('Position dans le range')).toBeNull();
    expect(screen.queryByText('Session')).toBeNull(); // no session windows → dropped
    // No fabricated « non disponible » placeholder is rendered as a tile value.
    expect(screen.queryByText('non disponible')).toBeNull();
  });

  it('when the tile count is odd, the last tile spans the full width', async () => {
    // marketStatus null drops Session; candles resolve → Levels present.
    const { container } = render(<Harness structure={FULL_STRUCT} marketStatus={null} />);
    await waitFor(() => expect(screen.getByText('Niveaux de référence')).toBeInTheDocument());
    const tiles = Array.from(container.querySelectorAll('.tile'));
    expect(tiles.length % 2).toBe(1); // odd
    expect(tiles[tiles.length - 1]!.classList.contains('span2')).toBe(true);
    // and no earlier tile is full-width
    for (const t of tiles.slice(0, -1)) expect(t.classList.contains('span2')).toBe(false);
  });

  it('opens Donnée from the tile and Concept from the « ? » (one panel at a time)', () => {
    render(<Harness structure={FULL_STRUCT} marketStatus={OPEN_STATUS} />);
    const volTile = screen.getByText('Volatilité').closest('.tile') as HTMLElement;

    // Click the tile body → Donnée tab, with the live ratio proof.
    fireEvent.click(volTile);
    expect(screen.getByText('Rapport')).toBeInTheDocument();
    expect(screen.getByText('0.95')).toBeInTheDocument();

    // Click the tile's « ? » → Concept tab, with its mandatory « ne dit pas » block.
    const help = within(volTile).getByLabelText('Expliquer cette mesure');
    fireEvent.click(help);
    expect(screen.getByText('Ce que ça ne dit pas')).toBeInTheDocument();
    // Donnée proof is no longer shown (single panel, switched to Concept).
    expect(screen.queryByText('Rapport')).toBeNull();
  });

  it('traces a reference level on click and removes it on re-click (dedicated channel)', async () => {
    const { container } = render(<Harness structure={FULL_STRUCT} marketStatus={OPEN_STATUS} />);
    await waitFor(() => expect(screen.getByText('Niveaux de référence')).toBeInTheDocument());
    fireEvent.click(screen.getByText('Niveaux de référence').closest('.tile') as HTMLElement);

    // The reference-level rows expose one clickable price button each; the first
    // is « Ouverture du jour » (day open = 2398.6).
    const dayOpenBtn = await waitFor(() => {
      const btns = container.querySelectorAll('.pxbtn');
      expect(btns.length).toBeGreaterThan(0);
      return btns[0] as HTMLElement;
    });
    expect(screen.getByTestId('ref')).toHaveTextContent('none');

    fireEvent.click(dayOpenBtn); // trace
    expect(screen.getByTestId('ref')).toHaveTextContent('Ouverture du jour=2398.6');
    expect(dayOpenBtn.getAttribute('aria-pressed')).toBe('true');

    fireEvent.click(dayOpenBtn); // re-click removes
    expect(screen.getByTestId('ref')).toHaveTextContent('none');
  });

  it('Session tile shows the current session, and its Donnée has no « État du marché » (no dup with the header badge)', () => {
    render(<Harness structure={FULL_STRUCT} marketStatus={OPEN_STATUS} />);
    // The tile is labelled « Session » (not « Horaire du marché ») and its value
    // is one of the session names — never the market state.
    const tile = screen.getByText('Session').closest('.tile') as HTMLElement;
    expect(tile).toBeTruthy();
    const value = tile.querySelector('.v')?.textContent ?? '';
    expect(['Asie', 'Londres', 'New York', 'Chevauchement Londres / NY', 'Hors session']).toContain(value);

    fireEvent.click(tile);
    // Donnée: session in progress + local time, and NOT the market state row.
    expect(screen.getByText('Session en cours')).toBeInTheDocument();
    expect(screen.getByText('Heure locale du marché')).toBeInTheDocument();
    expect(screen.queryByText('État du marché')).toBeNull();
  });
});

describe('RG-1 — the zone id-lock is not weakened by reference-level tracing', () => {
  it('coerceViewAction still rejects an invented (non-detected) zone id', () => {
    const valid = new Set<string>(['ob1', 'fvg1']);
    expect(
      coerceViewAction({ action: 'highlight_zone', params: { zone_id: 'INVENTED' } }, valid),
    ).toBeNull();
    // A real id still resolves.
    expect(
      coerceViewAction({ action: 'highlight_zone', params: { zone_id: 'ob1' } }, valid),
    ).not.toBeNull();
  });

  it('no ViewAction may carry a price/level (geometry stays out of the whitelist)', () => {
    const valid = new Set<string>(['ob1']);
    expect(
      coerceViewAction({ action: 'highlight_zone', params: { zone_id: 'ob1', level: 2400 } }, valid),
    ).toBeNull();
  });
});
