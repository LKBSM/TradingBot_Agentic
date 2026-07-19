import { fireEvent, render, screen } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import * as React from 'react';
import { ChartViewProvider, useChartView } from '../viewState';
import { coerceViewActions } from '../viewActions';

/**
 * NAV-05 — masks/isolation/highlight must be wiped when the combo CHANGES, but
 * preserved while it stays the same (so the /app↔/zones sharing survives).
 */
function Harness() {
  const { view, applyActions, resetForCombo } = useChartView();
  return (
    <div>
      <span data-testid="hidden">{view.hiddenZoneIds.join(',')}</span>
      <button
        data-testid="hide"
        onClick={() =>
          applyActions(
            coerceViewActions(
              [{ action: 'hide_zones', params: { zone_ids: ['z1'] } }],
              new Set(['z1']),
            ),
          )
        }
      >
        hide
      </button>
      <button data-testid="comboA" onClick={() => resetForCombo('XAUUSD:M15')}>
        A
      </button>
      <button data-testid="comboB" onClick={() => resetForCombo('EURUSD:H4')}>
        B
      </button>
    </div>
  );
}

describe('ChartViewProvider.resetForCombo', () => {
  it('keeps masks for the same combo, wipes them when the combo changes', () => {
    render(
      <ChartViewProvider>
        <Harness />
      </ChartViewProvider>,
    );
    // First combo observation (records the key, no wipe), then mask z1.
    fireEvent.click(screen.getByTestId('comboA'));
    fireEvent.click(screen.getByTestId('hide'));
    expect(screen.getByTestId('hidden').textContent).toBe('z1');

    // Same combo again → mask preserved (the intended /app↔/zones sharing).
    fireEvent.click(screen.getByTestId('comboA'));
    expect(screen.getByTestId('hidden').textContent).toBe('z1');

    // Combo change → stale mask wiped.
    fireEvent.click(screen.getByTestId('comboB'));
    expect(screen.getByTestId('hidden').textContent).toBe('');
  });
});
