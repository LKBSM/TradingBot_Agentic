import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type {
  FairValueGap,
  MarketReadingStructure,
  OrderBlock,
} from '@/types/market-reading';
import { StructureSection } from '../sections/StructureSection';
import { Accordion } from '@/components/ui/accordion';
import {
  ChartViewProvider,
  useChartView,
} from '@/lib/chart/viewState';

/**
 * Click-to-chart wiring (display/navigation only). Clicking a zone entry asks the
 * chart — through the SAME view channel the M.I.A Agent drives — to focus_zone
 * (centre) + highlight_zone (emphasise) that zone by its REAL engine id. These
 * tests assert: the right id is dispatched, an out-of-view zone gets a fresh
 * focus command (the chart turns that into a setVisibleRange), no invented id can
 * be dispatched, and the detected structure is never mutated.
 */

const mkOb = (i: number, over: Partial<OrderBlock> = {}): OrderBlock => ({
  id: `OB_bull_${i}`,
  level_low: 2300 + i,
  level_high: 2302 + i,
  importance: 'medium',
  status: 'active',
  created_at: '2026-06-20T10:00:00+00:00',
  tested: false,
  user_flagged: false,
  ...over,
});

const mkFvg = (i: number, over: Partial<FairValueGap> = {}): FairValueGap => ({
  id: `FVG_bull_${i}`,
  level_low: 2400 + i,
  level_high: 2402 + i,
  status: 'active',
  created_at: '2026-06-20T11:00:00+00:00',
  tested: false,
  user_flagged: false,
  ...over,
});

function structureWith(
  obs: OrderBlock[],
  fvgs: FairValueGap[] = [],
): MarketReadingStructure {
  return {
    bos: null,
    choch: null,
    order_blocks: obs,
    fair_value_gaps: fvgs,
    retest_in_progress: null,
  };
}

/** Surfaces the chart view state the click is supposed to drive. */
function ChartViewProbe() {
  const { view } = useChartView();
  return (
    <div>
      <span data-testid="focus-kind">{view.focus?.kind ?? 'none'}</span>
      <span data-testid="focus-zone">{view.focus?.zoneId ?? 'none'}</span>
      <span data-testid="focus-nonce">{view.focus?.nonce ?? 0}</span>
      <span data-testid="highlight">{view.highlightZoneId ?? 'none'}</span>
    </div>
  );
}

function renderWired(structure: MarketReadingStructure) {
  return render(
    <ChartViewProvider>
      <Accordion type="multiple" defaultValue={['structure']}>
        <StructureSection
          structure={structure}
          instrument="XAUUSD"
          closePrice={2305}
        />
      </Accordion>
      <ChartViewProbe />
    </ChartViewProvider>,
  );
}

describe('zone list → click to chart', () => {
  it('clicking an OB entry centres + highlights THAT zone by its real engine id', () => {
    renderWired(structureWith([mkOb(1)]));

    const entry = screen.getByRole('button', {
      name: /· actif/i,
    });
    fireEvent.click(entry);

    // focus_zone → a "zone" focus command carrying the real id (the chart turns
    // this into setVisibleRange); highlight_zone → the same id highlighted.
    expect(screen.getByTestId('focus-kind')).toHaveTextContent('zone');
    expect(screen.getByTestId('focus-zone')).toHaveTextContent('OB_bull_1');
    expect(screen.getByTestId('highlight')).toHaveTextContent('OB_bull_1');
    // The selected entry is marked (single source of truth = highlighted zone).
    expect(entry).toHaveAttribute('aria-pressed', 'true');
  });

  it('works for FVG entries too, using the real FVG id', () => {
    renderWired(structureWith([], [mkFvg(7)]));
    const entry = screen.getByRole('button', { name: /active/i });
    fireEvent.click(entry);
    expect(screen.getByTestId('focus-zone')).toHaveTextContent('FVG_bull_7');
    expect(screen.getByTestId('highlight')).toHaveTextContent('FVG_bull_7');
  });

  it('selecting another zone moves the selection (only one entry stays selected)', () => {
    renderWired(structureWith([mkOb(1), mkOb(2)]));
    const entries = screen.getAllByRole('button', {
      name: /· actif/i,
    });
    fireEvent.click(entries[0]!);
    expect(entries[0]!).toHaveAttribute('aria-pressed', 'true');
    expect(entries[1]!).toHaveAttribute('aria-pressed', 'false');
    const firstHighlight = screen.getByTestId('highlight').textContent;

    fireEvent.click(entries[1]!);
    expect(entries[0]!).toHaveAttribute('aria-pressed', 'false');
    expect(entries[1]!).toHaveAttribute('aria-pressed', 'true');
    // Highlight followed the click to the OTHER zone (id is order-independent —
    // the list re-ranks by proximity — so we assert it moved to a different one).
    const secondHighlight = screen.getByTestId('highlight').textContent;
    expect(secondHighlight).not.toBe(firstHighlight);
    expect(['OB_bull_1', 'OB_bull_2']).toContain(secondHighlight);
  });

  it('re-clicking the SAME zone deselects it (toggle: clears highlight + un-zooms)', () => {
    renderWired(structureWith([mkOb(1)]));
    const entry = screen.getByRole('button', {
      name: /· actif/i,
    });
    fireEvent.click(entry);
    expect(entry).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByTestId('highlight')).toHaveTextContent('OB_bull_1');
    const first = Number(screen.getByTestId('focus-nonce').textContent);

    fireEvent.click(entry);
    // Second click on the already-selected zone deselects it: the blue highlight
    // is cleared, the entry is no longer pressed, and the view un-zooms back to
    // recent price (focus kind 'price'). The nonce still bumps to re-frame.
    expect(entry).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByTestId('highlight')).toHaveTextContent('none');
    expect(screen.getByTestId('focus-kind')).toHaveTextContent('price');
    const second = Number(screen.getByTestId('focus-nonce').textContent);
    expect(second).toBeGreaterThan(first);
  });

  it('never dispatches an invented zone — only ids the engine emitted are focusable', () => {
    // The only clickable targets are the entries the engine emitted; each carries
    // a real id. After clicking, the focused id is always one of those ids.
    const structure = structureWith([mkOb(1)], [mkFvg(2)]);
    const emitted = new Set(['OB_bull_1', 'FVG_bull_2']);
    renderWired(structure);
    // `/·/` is the zone-entry separator — matches every OB ("actif") AND FVG
    // ("active") row, but not the « voir plus » / ⓘ buttons.
    for (const btn of screen.getAllByRole('button', { name: /·/ })) {
      fireEvent.click(btn);
      expect(emitted.has(screen.getByTestId('focus-zone').textContent ?? '')).toBe(
        true,
      );
    }
  });

  it('navigation never mutates the detected structure', () => {
    const structure = structureWith([mkOb(1), mkOb(2)], [mkFvg(3)]);
    const before = JSON.stringify(structure);
    renderWired(structure);
    for (const btn of screen.getAllByRole('button', { name: /·/ })) {
      fireEvent.click(btn);
    }
    // The engine's output is read-only here — click is display/navigation only.
    expect(JSON.stringify(structure)).toBe(before);
  });

  it('stays readable with no chart provider (entries render, click is a graceful no-op)', () => {
    // Outside the /app workspace there is no ChartViewProvider; the list must
    // still render and a click must not throw.
    render(
      <Accordion type="multiple" defaultValue={['structure']}>
        <StructureSection
          structure={structureWith([mkOb(1)])}
          instrument="XAUUSD"
          closePrice={2305}
        />
      </Accordion>,
    );
    const list = screen.getByRole('list');
    const entry = within(list).getByRole('button');
    expect(() => fireEvent.click(entry)).not.toThrow();
  });
});
