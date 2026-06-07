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
    expect(screen.getByText('Synthèse des conditions')).toBeInTheDocument();
  });

  it('wires the "Demander à Sentinel" CTA when a handler is provided', () => {
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
    expect(screen.getByText('trend')).toBeInTheDocument();
    expect(screen.getByText('Synthèse générée')).toBeInTheDocument();
  });

  it('marks template fallbacks distinctly', () => {
    render(
      <Accordion type="multiple" defaultValue={['conditions']}>
        <ConditionsSection conditions={FIXTURE_EUR_H1.conditions} />
      </Accordion>,
    );
    expect(screen.getByText(/Synthèse modèle/)).toBeInTheDocument();
  });
});
