import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Accordion } from '@/components/ui/accordion';
import { RegimeSection } from '../RegimeSection';
import type {
  MarketReadingHeader,
  MarketReadingRegime,
  MarketReadingStructure,
} from '@/types/market-reading';
import type { MtfTrendMap } from '@/lib/market-reading/mtf-trend';

// Mock the read-only MTF hook so the section renders deterministically without
// any network — the fetch wiring is covered separately; here we assert display.
const mockUseMtfTrends = vi.fn();
vi.mock('@/lib/market-reading/hooks', () => ({
  useMtfTrends: (instrument: string | null) => mockUseMtfTrends(instrument),
}));

const REGIME: MarketReadingRegime = {
  trend: 'bullish',
  volatility_observed: 'normal',
  market_phase: 'trend',
  mtf_confluence: {},
};

const HEADER: MarketReadingHeader = {
  instrument: 'XAUUSD',
  timeframe: 'M15',
  candle_close_ts: '2026-06-24T19:00:00',
  close_price: 2400,
};

// One active OB, one mitigated; two active FVG, one filled → « 1 OB · 2 FVG actifs ».
const STRUCTURE_FULL: MarketReadingStructure = {
  bos: null,
  choch: {
    direction: 'bullish',
    level: 2380,
    broken_at: '2026-06-24T14:30:00',
    validation_status: 'confirmed',
  },
  order_blocks: [
    { id: 'a', level_high: 2390, level_low: 2380, importance: 'high', status: 'active', created_at: '2026-06-24T10:00:00', tested: false, user_flagged: false },
    { id: 'b', level_high: 2370, level_low: 2360, importance: 'low', status: 'mitigated', created_at: '2026-06-24T09:00:00', tested: true, user_flagged: false },
  ],
  fair_value_gaps: [
    { id: 'c', level_high: 2395, level_low: 2392, status: 'active', created_at: '2026-06-24T11:00:00', tested: false, user_flagged: false },
    { id: 'd', level_high: 2388, level_low: 2385, status: 'active', created_at: '2026-06-24T11:30:00', tested: false, user_flagged: false },
    { id: 'e', level_high: 2350, level_low: 2348, status: 'filled', created_at: '2026-06-24T08:00:00', tested: true, user_flagged: false },
  ],
};

const STRUCTURE_EMPTY: MarketReadingStructure = {
  bos: null,
  choch: null,
  order_blocks: [],
  fair_value_gaps: [],
};

function renderSection(
  trends: MtfTrendMap,
  {
    isLoading = false,
    structure = STRUCTURE_FULL,
    regime = REGIME,
    header = HEADER,
  }: {
    isLoading?: boolean;
    structure?: MarketReadingStructure;
    regime?: MarketReadingRegime;
    header?: MarketReadingHeader;
  } = {},
) {
  mockUseMtfTrends.mockReturnValue({ trends, isLoading });
  return render(
    <Accordion type="multiple" defaultValue={['regime']}>
      <RegimeSection regime={regime} structure={structure} header={header} />
    </Accordion>,
  );
}

beforeEach(() => mockUseMtfTrends.mockReset());

describe('RegimeSection — header + existing facts', () => {
  it('keeps the panel title and the volatility badge', () => {
    renderSection({ h4: 'bullish', h1: 'bullish', m15: 'bullish' });
    expect(screen.getByText('Régime de marché')).toBeInTheDocument();
    expect(screen.getByText('Volatilité normale')).toBeInTheDocument();
  });

  it('shows the three timeframes at a glance with arrows', () => {
    renderSection({ h4: 'bullish', h1: 'bullish', m15: 'bearish' });
    expect(screen.getByText(/H4\s*↗/)).toBeInTheDocument();
    expect(screen.getByText(/H1\s*↗/)).toBeInTheDocument();
    expect(screen.getByText(/M15\s*↘/)).toBeInTheDocument();
  });

  it('keeps a descriptive (non-instruction) disclaimer', () => {
    renderSection({ h4: 'neutral', h1: 'neutral', m15: 'neutral' });
    expect(
      screen.getByText(/ne constitue.* pas une instruction adressée au trader/i),
    ).toBeInTheDocument();
  });

  it('degrades gracefully when no timeframe is available', () => {
    renderSection({ h4: null, h1: null, m15: null });
    expect(
      screen.getByText('Alignement multi-timeframe indisponible.'),
    ).toBeInTheDocument();
  });
});

describe('RegimeSection — (a) market phase', () => {
  it('shows the compact phase badge next to the regime', () => {
    renderSection({ h4: 'bullish', h1: 'bullish', m15: 'bullish' });
    expect(screen.getByText('Phase : Tendance')).toBeInTheDocument();
  });
});

describe('RegimeSection — (b) trend maturity, (c) last event, (d) zone density', () => {
  it('renders the real engine data for each fact', () => {
    renderSection({ h4: 'bullish', h1: 'bullish', m15: 'bullish' });
    expect(
      screen.getByText(
        'Structure orientée haussière depuis le CHOCH du 24/06 à 14:30 (≈ 18 bougies M15).',
      ),
    ).toBeInTheDocument();
    expect(screen.getByText('CHOCH haussier confirmé (M15)')).toBeInTheDocument();
    expect(screen.getByText('1 OB · 2 FVG actifs')).toBeInTheDocument();
  });

  it('shows « non disponible » for missing facts, never invents them', () => {
    renderSection({ h4: 'bullish', h1: 'bullish', m15: 'bullish' }, {
      structure: STRUCTURE_EMPTY,
    });
    // Maturity + last event have no engine datum → « non disponible » (×2).
    expect(screen.getAllByText('non disponible')).toHaveLength(2);
    // Density is always a real fact, even at zero.
    expect(screen.getByText('0 OB · 0 FVG actifs')).toBeInTheDocument();
  });
});

describe('RegimeSection — (e) multi-TF disagreement', () => {
  it('shows a distinct warn callout when a TF goes against the others', () => {
    renderSection({ h4: 'bullish', h1: 'bullish', m15: 'bearish' });
    expect(screen.getByText('Désaccord multi-timeframe')).toBeInTheDocument();
    expect(
      screen.getByText('M15 se replie contre la tendance H4 haussière.'),
    ).toBeInTheDocument();
  });

  it('does NOT show the disagreement callout when the TFs are aligned', () => {
    renderSection({ h4: 'bullish', h1: 'bullish', m15: 'bullish' });
    expect(screen.queryByText('Désaccord multi-timeframe')).not.toBeInTheDocument();
    expect(screen.getByText('Les 3 TF sont alignés (haussiers).')).toBeInTheDocument();
  });
});

describe('RegimeSection — no predictive output', () => {
  it('renders no probabilistic / directive vocabulary', () => {
    const { container } = renderSection({
      h4: 'bullish',
      h1: 'bullish',
      m15: 'bearish',
    });
    const text = container.textContent ?? '';
    for (const re of [/probab/i, /\d+\s*%/, /objectif/i, /acheter/i, /vendre/i]) {
      expect(text).not.toMatch(re);
    }
  });
});
