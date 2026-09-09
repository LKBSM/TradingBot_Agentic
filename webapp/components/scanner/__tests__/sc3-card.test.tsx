import { render as rtlRender, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import { ComboCard } from '../ComboCard';
import type { ComboMatch } from '@/lib/conditions/types';
import messages from '@/messages/fr.json';

/**
 * SC-3 — the visual density pass on the scanner card.
 *
 * Two things are asserted here, and they pull in opposite directions on purpose:
 *   1. HONESTY — no chip may appear without a real field behind it, and no fact
 *      may be lost to the fold. Every met condition is still rendered; the
 *      context block still holds every line it held before.
 *   2. NON-REGRESSION on the one promise this mission is NOT allowed to trade
 *      for density: « ce qui va à l'encontre » is never collapsible.
 */
function render(ui: React.ReactElement) {
  return rtlRender(
    <NextIntlClientProvider locale="fr" messages={messages}>
      {ui}
    </NextIntlClientProvider>,
  );
}

function makeMatch(overrides: Partial<ComboMatch> = {}): ComboMatch {
  return {
    instrument: 'XAUUSD',
    timeframe: 'H4',
    candle_close_ts: '2026-05-28T14:00:00+00:00',
    close_price: 4408.91702,
    matched: true,
    met_count: 1,
    total: 1,
    conditions_met: [
      { type: 'trend_is', label: 'La tendance structurelle est', met: true, detail: 'haussier.' },
    ],
    conditions_unmet: [],
    context: {
      trend: 'bullish',
      market_phase: 'expansion',
      volatility_observed: 'normal',
      mtf_confluence: {},
      mtf_trends: {},
      bos: { direction: 'bearish', level: 4380, validation_status: 'confirmed' },
      choch: null,
      active_order_blocks: 4,
      active_fair_value_gaps: 10,
      structural_range: { low: 4224.23, high: 4695.94 },
      news_upcoming: [],
    },
    ...overrides,
  };
}

function chipKeys(container: HTMLElement): string[] {
  return Array.from(container.querySelectorAll('.chip')).map(
    (n) => n.getAttribute('data-chip') ?? '',
  );
}

describe('SC-3 — block 2 never folds', () => {
  it('renders « ce qui va à l\'encontre » outside any <details>', () => {
    const { container } = render(<ComboCard match={makeMatch()} locale="fr" />);
    const block = screen.getByTestId('against-block');
    expect(block).toBeInTheDocument();
    // The promise is structural, not a matter of the default open state: the
    // block must have NO <details> ancestor at all, inside this card.
    expect(block.closest('details')).toBeNull();
    // …and the card does fold something, so this is a real distinction.
    expect(container.querySelectorAll('details').length).toBeGreaterThan(0);
  });

  it('keeps the against block, and its lines, on a FULL match', () => {
    const match = makeMatch({
      context_against: [
        { label: 'Le 5 min, 15 min, 1 h', detail: 'en tendance baissière — désaccord multi-unités' },
      ],
    });
    render(<ComboCard match={match} locale="fr" />);
    expect(screen.getByTestId('against-block').closest('details')).toBeNull();
    expect(screen.getByText(/désaccord multi-unités/)).toBeInTheDocument();
  });
});

describe('SC-3 — block 3 folds, and loses nothing', () => {
  it('is a <details> closed by default', () => {
    render(<ComboCard match={makeMatch()} locale="fr" />);
    const details = screen.getByTestId('context-block') as HTMLDetailsElement;
    expect(details.tagName).toBe('DETAILS');
    expect(details.open).toBe(false);
  });

  it('keeps every context fact in the DOM while folded', () => {
    const { container } = render(<ComboCard match={makeMatch()} locale="fr" />);
    const body = container.querySelector('.ctxd .ctx2')?.textContent ?? '';
    expect(body).toMatch(/4 OB/);
    expect(body).toMatch(/10 FVG/);
    expect(body).toMatch(/4224\.23–4695\.94/);
  });

  it('carries the high-impact news count in the summary so it reads folded', () => {
    const match = makeMatch({
      context: {
        ...makeMatch().context,
        news_upcoming: [
          { event: 'US NFP', impact: 'high', time_to_event_min: 30 },
          { event: 'US CPI', impact: 'high', time_to_event_min: 90 },
          { event: 'EU PMI', impact: 'medium', time_to_event_min: 45 },
        ],
      },
    });
    const { container } = render(<ComboCard match={match} locale="fr" />);
    const summary = container.querySelector('.ctxd > summary')?.textContent ?? '';
    // Only the HIGH-impact ones are counted (the medium is noise, as before).
    expect(summary).toMatch(/2 actus importantes/);
  });

  it('shows the plain summary when no high-impact news is upcoming', () => {
    const { container } = render(<ComboCard match={makeMatch()} locale="fr" />);
    const summary = container.querySelector('.ctxd > summary')?.textContent ?? '';
    // The bundles use the TYPOGRAPHIC apostrophe (U+2019), never the straight one.
    expect(summary).toBe(messages.scanner.combo.contextBlock);
    expect(summary).not.toMatch(/actus? importante/);
  });

  it('the folded summary is the block label plus the count, in all 9 locales', async () => {
    // Guards the pair: `contextBlockWithNews` must READ as the block label with
    // the news count appended — a translator changing one must change the other.
    const locales = ['ar', 'de', 'en', 'es', 'fr', 'it', 'nl', 'pl', 'pt'];
    for (const loc of locales) {
      const bundle = (await import(`@/messages/${loc}.json`)).default;
      const combo = bundle.scanner.combo;
      expect(combo.contextBlockWithNews.startsWith(combo.contextBlock)).toBe(true);
      for (const k of ['chipsAria', 'chipOb', 'chipFvg', 'chipBos']) {
        expect(typeof combo[k]).toBe('string');
        expect(combo[k].length).toBeGreaterThan(0);
      }
    }
  });
});

describe('SC-3 — chips are backed by real fields only', () => {
  it('renders one chip per present field, in a fixed order', () => {
    const { container } = render(<ComboCard match={makeMatch()} locale="fr" />);
    expect(chipKeys(container)).toEqual(['trend', 'phase', 'ob', 'fvg', 'bos', 'range']);
  });

  it('drops the range chip when structural_range is null (no dash, no zero)', () => {
    const match = makeMatch({ context: { ...makeMatch().context, structural_range: null } });
    const { container } = render(<ComboCard match={match} locale="fr" />);
    expect(chipKeys(container)).not.toContain('range');
    expect(container.querySelector('.chips')?.textContent).not.toMatch(/—|–\s*$/);
  });

  it('drops the BOS chip when no break is reported', () => {
    const match = makeMatch({ context: { ...makeMatch().context, bos: null } });
    const { container } = render(<ComboCard match={match} locale="fr" />);
    expect(chipKeys(container)).not.toContain('bos');
  });

  it('drops a zone chip on a zero count rather than showing « 0 OB »', () => {
    const match = makeMatch({
      context: { ...makeMatch().context, active_order_blocks: 0, active_fair_value_gaps: 0 },
    });
    const { container } = render(<ComboCard match={match} locale="fr" />);
    expect(chipKeys(container)).toEqual(['trend', 'phase', 'bos', 'range']);
    expect(container.querySelector('.chips')?.textContent).not.toMatch(/\b0 (OB|FVG)/);
    // The exact counts are NOT lost — the context block still states them.
    expect(container.querySelector('.ctxd .ctx2')?.textContent).toMatch(/0 OB/);
  });

  it('renders no chip row at all when the context carries nothing', () => {
    const match = makeMatch({
      context: {
        trend: null,
        market_phase: null,
        volatility_observed: null,
        mtf_confluence: {},
        bos: null,
        choch: null,
        active_order_blocks: 0,
        active_fair_value_gaps: 0,
        structural_range: null,
        news_upcoming: [],
      },
    });
    const { container } = render(<ComboCard match={match} locale="fr" />);
    expect(container.querySelector('.chips')).toBeNull();
  });

  it('colours the dots from the theme tokens, never a hard-coded hex', () => {
    const { container } = render(<ComboCard match={makeMatch()} locale="fr" />);
    const tones = Array.from(container.querySelectorAll('.chip .dot')).map((n) => n.className);
    expect(tones).toEqual([
      'dot d-bull', // trend bullish
      'dot d-neutral', // phase
      'dot d-ob',
      'dot d-fvg',
      'dot d-bear', // BOS bearish
      'dot d-acc', // structural range
    ]);
    // No inline colour anywhere in the row: the four themes drive it via CSS.
    expect(container.querySelector('.chips')?.innerHTML).not.toMatch(/style="[^"]*color/);
  });
});

describe('SC-3 — block 1 condenses without amputating', () => {
  it('renders EVERY met condition, the first one leading', () => {
    const match = makeMatch({
      met_count: 3,
      total: 3,
      conditions_met: [
        { type: 'trend_is', label: 'La tendance structurelle est', met: true, detail: 'haussier.' },
        { type: 'market_phase_is', label: 'La phase de marché est', met: true, detail: 'expansion.' },
        {
          type: 'price_in_ob',
          label: 'Le prix est dans un Order Block',
          met: true,
          detail: 'Prix dans un Order Block actif (4382.1–4401.7).',
        },
      ],
    });
    const { container } = render(<ComboCard match={match} locale="fr" />);
    const lines = Array.from(container.querySelectorAll('.cl.yes'));
    expect(lines).toHaveLength(3);
    expect(lines[0]?.className).toContain('lead');
    expect(lines[1]?.className).toContain('sub');
    expect(lines[2]?.className).toContain('sub');
    // The third one keeps the em-dash form — its detail is a sentence.
    expect(lines[2]?.textContent).toContain('Le prix est dans un Order Block — Prix dans un');
  });

  it('keeps the denominator and the non-evaluable count intact', () => {
    const match = makeMatch({ met_count: 2, total: 3, non_evaluable_count: 1 });
    render(<ComboCard match={match} locale="fr" />);
    expect(screen.getByText(/2 de tes 3 conditions/)).toBeInTheDocument();
    expect(screen.getByText(/1 non évaluable/)).toBeInTheDocument();
  });
});
