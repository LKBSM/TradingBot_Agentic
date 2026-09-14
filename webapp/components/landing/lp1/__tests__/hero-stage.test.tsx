import { render as rtlRender, screen, fireEvent, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import fr from '@/messages/fr.json';
import { HeroStage } from '../HeroStage';

/**
 * LP-3 §B — the hero's arrival sequence.
 *
 * Three promises are pinned here, because all three are easy to break without
 * noticing on a page nobody re-reads:
 *
 *   1. `prefers-reduced-motion: reduce` gets the FINISHED window, immediately —
 *      not a slower animation, not a queued one.
 *   2. the sequence never holds the interaction hostage: a click on a layer
 *      during the arrival stops it AND is applied.
 *   3. no figure is animated: every number is at its final value on the very
 *      first (server) render.
 */

function stubReducedMotion(reduce: boolean) {
  vi.stubGlobal(
    'matchMedia',
    (query: string) =>
      ({
        matches: reduce && query.includes('prefers-reduced-motion'),
        media: query,
        onchange: null,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }) as unknown as MediaQueryList,
  );
}

function render() {
  return rtlRender(
    <NextIntlClientProvider locale="fr" messages={fr}>
      <HeroStage />
    </NextIntlClientProvider>,
  );
}

const stage = () => document.querySelector('[data-seq]') as HTMLElement;

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('LP-3 hero — reduced motion gets the final state, with no animation', () => {
  beforeEach(() => stubReducedMotion(true));

  it('settles immediately: no step is ever scheduled', () => {
    vi.useFakeTimers();
    const setTimeoutSpy = vi.spyOn(window, 'setTimeout');
    render();
    // `done` is reached on mount — not after a tick, not after the arrival.
    expect(stage()).toHaveAttribute('data-seq', 'done');
    expect(setTimeoutSpy).not.toHaveBeenCalled();
  });

  it('renders the FINISHED window: full narration, both chat bubbles, status', () => {
    render();
    // the whole reading, all four families — not a truncated typewriter frame
    expect(screen.getByText(/CHOCH haussier/)).toBeInTheDocument();
    expect(screen.getByText(/Order Block haussier/)).toBeInTheDocument();
    expect(screen.getByText(/Fair Value Gap baissier comblé/)).toBeInTheDocument();
    expect(screen.getByText(/liquidité achat reste intacte/)).toBeInTheDocument();
    // the exchange that ends on a refusal is fully there
    expect(screen.getByText('tu penses que ça va rebondir ?')).toBeInTheDocument();
    expect(screen.getByText(/Je ne réponds pas à ça/)).toBeInTheDocument();
    // and the settled status label
    expect(screen.getByText('lecture à jour')).toBeInTheDocument();
  });
});

describe('LP-3 hero — the visitor is never locked out', () => {
  beforeEach(() => stubReducedMotion(false));

  it('a click on a layer during the arrival stops it AND is applied', () => {
    render();
    // the arrival is running: the window has not settled
    expect(stage()).toHaveAttribute('data-seq', 'idle');
    // mid-sequence, the visitor unticks a layer — the chips are live already
    const chip = screen.getByRole('button', { name: 'Fair Value Gaps' });
    fireEvent.pointerDown(chip);
    fireEvent.click(chip);
    // 1. the sequence yielded
    expect(stage()).toHaveAttribute('data-seq', 'done');
    // 2. the click itself went through: the FVG fragment left the narration…
    expect(screen.queryByText(/Fair Value Gap baissier comblé/)).not.toBeInTheDocument();
    // …while the layers the visitor kept are still described
    expect(screen.getByText(/Order Block haussier/)).toBeInTheDocument();
    expect(chip).toHaveAttribute('aria-pressed', 'false');
  });

  it('once stopped, a timer still in flight cannot restart the arrival', () => {
    vi.useFakeTimers();
    render();
    fireEvent.pointerDown(screen.getByRole('button', { name: 'Liquidité' }));
    expect(stage()).toHaveAttribute('data-seq', 'done');
    // let every scheduled step fire into the void
    act(() => { vi.advanceTimersByTime(15_000); });
    expect(stage()).toHaveAttribute('data-seq', 'done');
    // the reading is the real composed one, not a typewriter frame
    expect(screen.getByText(/CHOCH haussier/)).toBeInTheDocument();
  });

  it('the chart is interactive from the very first render', () => {
    render();
    for (const name of ['Order Blocks', 'Fair Value Gaps', 'Liquidité', 'BOS / CHOCH']) {
      const b = screen.getByRole('button', { name });
      expect(b).toBeEnabled();
      expect(b).toHaveAttribute('aria-pressed', 'true');
    }
  });
});

describe('LP-3 hero — no animated counter', () => {
  beforeEach(() => stubReducedMotion(false));

  it('every figure is at its final value on the first render', () => {
    render();
    const txt = document.body.textContent ?? '';
    // the price tag and the detected levels, verbatim — nothing counts up
    expect(txt).toContain('4 026,77');
    expect(txt).toContain('4 026,80');
    expect(txt).toContain('4 028,90');
    expect(txt).toContain('4 021,45');
  });

  it('the three status labels are all in the DOM; only the settled one is readable', () => {
    render();
    expect(screen.getByText('Le graphique se charge')).toHaveAttribute('aria-hidden', 'true');
    expect(screen.getByText('M.I.A lit la structure')).toHaveAttribute('aria-hidden', 'true');
    expect(screen.getByText('lecture à jour')).not.toHaveAttribute('aria-hidden');
  });
});
