import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { MarketPhasePanel } from '../MarketPhasePanel';
import { MarketReadingCard } from '../MarketReadingCard';
import { MarketReadingHeader } from '../MarketReadingHeader';
import { StructureSection } from '../sections/StructureSection';
import { ConditionsSection } from '../sections/ConditionsSection';
import { Accordion } from '@/components/ui/accordion';
import {
  FIXTURE_EUR_H1,
  FIXTURE_QUIET_XAU_H4,
  FIXTURE_XAU_M15,
} from '@/lib/market-reading/fixtures';

describe('MarketPhasePanel (ex-ConvictionGauge)', () => {
  it('renders three descriptive phase badges', () => {
    render(<MarketPhasePanel regime={FIXTURE_XAU_M15.regime} />);
    expect(screen.getByText('Tendance haussière')).toBeInTheDocument();
    expect(screen.getByText('Volatilité normale')).toBeInTheDocument();
    expect(screen.getByText('Phase de tendance')).toBeInTheDocument();
  });

  it('never renders a 0-100 conviction score', () => {
    const { container } = render(
      <MarketPhasePanel regime={FIXTURE_XAU_M15.regime} />,
    );
    // No "/ 100" gauge text, no role=meter (the old ConvictionGauge artefacts).
    expect(container.textContent).not.toContain('/ 100');
    expect(container.querySelector('[role="meter"]')).toBeNull();
    expect(screen.getByRole('group', { name: /phase de marché/i })).toBeInTheDocument();
  });

  it('reflects a ranging / low-vol regime', () => {
    render(<MarketPhasePanel regime={FIXTURE_EUR_H1.regime} />);
    expect(screen.getByText('Marché en range')).toBeInTheDocument();
    expect(screen.getByText('Volatilité basse')).toBeInTheDocument();
  });
});

describe('MarketReadingHeader', () => {
  it('shows instrument, timeframe and close price', () => {
    render(<MarketReadingHeader header={FIXTURE_XAU_M15.header} />);
    expect(screen.getByText('Or (XAU/USD)')).toBeInTheDocument();
    expect(screen.getByText(/XAUUSD · 15 minutes/)).toBeInTheDocument();
    expect(screen.getByText(/2[\s ]?392,35/)).toBeInTheDocument();
  });

  it('shows the candle-close relative age after mount', async () => {
    render(<MarketReadingHeader header={FIXTURE_XAU_M15.header} />);
    expect(await screen.findByText(/Bougie clôturée/)).toBeInTheDocument();
  });

  it('shows the unified live price + daily change when `live` is supplied', () => {
    // A unified price that DIFFERS from the per-timeframe close_price (the bug:
    // M15 vs H1/H4 divergence) — the header must show the unified one.
    render(
      <MarketReadingHeader
        header={FIXTURE_XAU_M15.header}
        live={{
          price: 4131.4,
          priceTs: 0,
          referenceClose: 4268.7,
          changeAbs: -137.3,
          changePct: -0.0322,
        }}
      />,
    );
    expect(screen.getByText(/4[\s ]?131,40/)).toBeInTheDocument();
    expect(screen.getByText(/−3,22 %/)).toBeInTheDocument();
    // The per-timeframe close_price must NOT also be shown.
    expect(screen.queryByText(/2[\s ]?392,35/)).not.toBeInTheDocument();
  });

  it('falls back to close_price when no live price is available', () => {
    render(<MarketReadingHeader header={FIXTURE_XAU_M15.header} live={null} />);
    expect(screen.getByText(/2[\s ]?392,35/)).toBeInTheDocument();
  });
});

describe('MarketReadingCard', () => {
  it('renders the hero + collapsible sections from a MarketReading', () => {
    render(<MarketReadingCard reading={FIXTURE_XAU_M15} />);
    expect(screen.getByText('Or (XAU/USD)')).toBeInTheDocument();
    expect(screen.getByText('Tendance haussière')).toBeInTheDocument();
    // Section triggers present (Layer 2).
    expect(screen.getByText('Structure de marché')).toBeInTheDocument();
    expect(screen.getByText('Régime de marché')).toBeInTheDocument();
    expect(screen.getByText('Contexte événementiel')).toBeInTheDocument();
    expect(screen.getByText('Lecture narrée')).toBeInTheDocument();
  });

  it('wires the "Demander à M.I.A Agent" CTA when a handler is provided', () => {
    const onAsk = vi.fn();
    render(<MarketReadingCard reading={FIXTURE_XAU_M15} onAskChatbot={onAsk} />);
    const cta = screen.getByRole('button', {
      name: /ouvrir le chatbot/i,
    });
    expect(cta).toBeEnabled();
    cta.click();
    expect(onAsk).toHaveBeenCalledOnce();
  });

  it('disables the CTA when no handler is provided', () => {
    render(<MarketReadingCard reading={FIXTURE_XAU_M15} />);
    expect(
      screen.getByRole('button', { name: /ouvrir le chatbot/i }),
    ).toBeDisabled();
  });

  it('hides the sections in heroOnly mode', () => {
    render(<MarketReadingCard reading={FIXTURE_XAU_M15} heroOnly />);
    expect(screen.queryByText('Structure de marché')).not.toBeInTheDocument();
  });
});

describe('StructureSection', () => {
  it('renders SMC facts when present', () => {
    render(
      <Accordion type="multiple" defaultValue={['structure']}>
        <StructureSection
          structure={FIXTURE_XAU_M15.structure}
          instrument="XAUUSD"
        />
      </Accordion>,
    );
    // BOS line — match on its price to disambiguate from the CHOCH line
    // (both are bullish + confirmed in the fixture).
    expect(
      screen.getByText(/2[\s ]?391,50 · haussier · confirmée/),
    ).toBeInTheDocument();
  });

  it('exposes a vulgarisation tooltip trigger for SMC terms (5.D)', () => {
    render(
      <Accordion type="multiple" defaultValue={['structure']}>
        <StructureSection
          structure={FIXTURE_XAU_M15.structure}
          instrument="XAUUSD"
        />
      </Accordion>,
    );
    // Each SMC label becomes an accessible InfoTooltip button (term + ⓘ).
    expect(
      screen.getByRole('button', {
        name: /Cassure de structure \(BOS\) — définition/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /Order Block — définition/i }),
    ).toBeInTheDocument();
  });

  it('does not contradict itself: a BOS retest with a now-stale break (no "aucune cassure")', () => {
    // Founder eval 2026-06-08: the engine emits `bos` only on a FRESH break, so
    // bars later `bos` is null while a `bos_retest` is armed. The BOS row must
    // not claim "aucune cassure récente" while the retest row shows a BOS retest.
    const structure = {
      bos: null,
      choch: null,
      order_blocks: [],
      fair_value_gaps: [],
      retest_in_progress: {
        level: 2391.5,
        type: 'bos_retest' as const,
        started_at: '2026-05-26T11:30:00+00:00',
      },
    };
    render(
      <Accordion type="multiple" defaultValue={['structure']}>
        <StructureSection structure={structure} instrument="XAUUSD" />
      </Accordion>,
    );
    expect(screen.queryByText('aucune cassure récente')).not.toBeInTheDocument();
    expect(
      screen.getByText(/cassure antérieure en cours de retest/),
    ).toBeInTheDocument();
    // The retest row still surfaces the live retest.
    expect(
      screen.getByText(/retest de cassure \(BOS\)/),
    ).toBeInTheDocument();
  });

  it('shows the present-tense editorial framing (1a) in French, no English/directive wording', () => {
    render(
      <Accordion type="multiple" defaultValue={['structure']}>
        <StructureSection
          structure={FIXTURE_XAU_M15.structure}
          instrument="XAUUSD"
        />
      </Accordion>,
    );
    const framing = screen.getByText(/Structures décrites au présent/);
    expect(framing).toBeInTheDocument();
    expect(framing.textContent).toMatch(/reprise ou invalidation/);
    expect(framing.textContent).toMatch(/Order Block et Fair Value Gap/);
    // No English leak, no directive verb.
    expect(framing.textContent).not.toMatch(/\b(shown|while|until|invalidated|buy|sell)\b/i);
  });

  it('shows an empty-state line when structure is bare', () => {
    render(
      <Accordion type="multiple" defaultValue={['structure']}>
        <StructureSection
          structure={FIXTURE_QUIET_XAU_H4.structure}
          instrument="XAUUSD"
        />
      </Accordion>,
    );
    expect(
      screen.getByText(/Aucun élément structurel notable/),
    ).toBeInTheDocument();
  });
});

describe('ConditionsSection', () => {
  it('renders the synthesis, tags and the source label', () => {
    render(
      <Accordion type="multiple" defaultValue={['conditions']}>
        <ConditionsSection conditions={FIXTURE_XAU_M15.conditions} />
      </Accordion>,
    );
    expect(
      screen.getByText(/Structure haussière confirmée/),
    ).toBeInTheDocument();
    // Tags are vulgarised for display (5.D): raw `trend` → "Tendance établie",
    // `retest_active` → "Retest en cours" (the snake_case code stays backend-side).
    expect(screen.getByText('Tendance établie')).toBeInTheDocument();
    expect(screen.getByText('Retest en cours')).toBeInTheDocument();
    expect(screen.getByText('Narration générée')).toBeInTheDocument();
  });

  it('marks template fallbacks distinctly', () => {
    render(
      <Accordion type="multiple" defaultValue={['conditions']}>
        <ConditionsSection conditions={FIXTURE_EUR_H1.conditions} />
      </Accordion>,
    );
    expect(screen.getByText(/Lecture modèle/)).toBeInTheDocument();
  });
});
