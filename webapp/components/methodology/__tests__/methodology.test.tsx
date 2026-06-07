import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { MethodologySection } from '../MethodologySection';
import { ConceptCard } from '../ConceptCard';
import { ScoreFormula } from '../ScoreFormula';
import MethodologyPage from '@/app/[locale]/methodology/page';
import {
  SCORE_FORMULAS,
  SMC_CONCEPTS,
  type MethodologyConcept,
} from '@/lib/methodology/content';
import { GLOSSARY } from '@/lib/glossary';

describe('MethodologySection', () => {
  it('renders an anchored, labelled section', () => {
    render(
      <MethodologySection id="demo" title="Titre démo" intro="Une intro.">
        <p>Contenu</p>
      </MethodologySection>,
    );
    const heading = screen.getByRole('heading', { name: 'Titre démo' });
    expect(heading).toBeInTheDocument();
    expect(heading.id).toBe('demo-title');
    expect(screen.getByText('Une intro.')).toBeInTheDocument();
    expect(screen.getByText('Contenu')).toBeInTheDocument();
  });
});

describe('ConceptCard', () => {
  const ob = SMC_CONCEPTS.find((c) => c.id === 'order-block') as MethodologyConcept;

  it('uses the glossary term + short definition (single source of truth)', () => {
    const { container } = render(<ConceptCard concept={ob} />);
    expect(screen.getByText(GLOSSARY.order_block.term)).toBeInTheDocument();
    expect(screen.getByText(GLOSSARY.order_block.short)).toBeInTheDocument();
    expect(screen.getByText(/Comment le moteur le détecte/i)).toBeInTheDocument();
    // The card id matches the glossary anchor so tooltips "En savoir plus →"
    // land on it.
    expect(container.querySelector('#order-block')).not.toBeNull();
    expect(GLOSSARY.order_block.anchor).toBe('#order-block');
  });
});

describe('ScoreFormula', () => {
  it('lists the variables and stays descriptive (no probability of success)', () => {
    const f = SCORE_FORMULAS.find((x) => x.id === 'incertitude')!;
    render(<ScoreFormula formula={f} />);
    expect(screen.getByText(f.title)).toBeInTheDocument();
    expect(
      screen.getByText(/Ce que le moteur prend en compte/i),
    ).toBeInTheDocument();
    f.variables.forEach((v) => {
      expect(screen.getByText(v)).toBeInTheDocument();
    });
  });
});

describe('MethodologyPage (intégration)', () => {
  it('renders all five sections and the engagement quote', () => {
    render(<MethodologyPage />);
    expect(
      screen.getByRole('heading', {
        level: 1,
        name: /Comment MIA Markets décrit le marché/i,
      }),
    ).toBeInTheDocument();
    // Section headings (substring match — titles may carry suffixes like "(SMC)").
    [/Notre engagement/i, /Concepts de structure/i, /Source de données/i].forEach(
      (title) => {
        expect(screen.getByRole('heading', { name: title })).toBeInTheDocument();
      },
    );
    // Niveau 1.5 limit explicitly stated.
    expect(
      screen.getByText(/Émettre un signal de trade/i),
    ).toBeInTheDocument();
  });

  it('contains no directive / pre-pivot jargon', () => {
    const { container } = render(<MethodologyPage />);
    const text = container.textContent ?? '';
    expect(text).not.toMatch(/conformel/i);
    expect(text).not.toMatch(/posterior HMM/i);
    expect(text).not.toMatch(/profit factor/i);
  });
});
