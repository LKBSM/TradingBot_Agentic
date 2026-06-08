import { renderHook, act } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import * as React from 'react';
import {
  ActiveComboProvider,
  comboKey,
  sameCombo,
  SUPPORTED_COMBOS,
  useActiveCombo,
} from '../store';

describe('active-combo store', () => {
  it('exposes the 6 supported combos (XAUUSD/EURUSD × M15/H1/H4)', () => {
    expect(SUPPORTED_COMBOS).toHaveLength(6);
    expect(SUPPORTED_COMBOS.map(comboKey)).toEqual([
      'XAUUSD:M15',
      'XAUUSD:H1',
      'XAUUSD:H4',
      'EURUSD:M15',
      'EURUSD:H1',
      'EURUSD:H4',
    ]);
  });

  it('compares combos structurally', () => {
    expect(
      sameCombo(
        { instrument: 'XAUUSD', timeframe: 'M15' },
        { instrument: 'XAUUSD', timeframe: 'M15' },
      ),
    ).toBe(true);
    expect(
      sameCombo(
        { instrument: 'XAUUSD', timeframe: 'M15' },
        { instrument: 'XAUUSD', timeframe: 'H1' },
      ),
    ).toBe(false);
    expect(sameCombo(null, null)).toBe(true);
    expect(sameCombo(null, { instrument: 'XAUUSD', timeframe: 'M15' })).toBe(
      false,
    );
  });

  it('selects and clears the active combo through the provider', () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <ActiveComboProvider>{children}</ActiveComboProvider>
    );
    const { result } = renderHook(() => useActiveCombo(), { wrapper });

    expect(result.current.active).toBeNull();
    expect(result.current.combos).toHaveLength(6);

    act(() => result.current.select({ instrument: 'EURUSD', timeframe: 'H1' }));
    expect(result.current.active).toEqual({
      instrument: 'EURUSD',
      timeframe: 'H1',
    });

    act(() => result.current.select(null));
    expect(result.current.active).toBeNull();
  });

  it('honours the initial combo', () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <ActiveComboProvider initial={{ instrument: 'XAUUSD', timeframe: 'M15' }}>
        {children}
      </ActiveComboProvider>
    );
    const { result } = renderHook(() => useActiveCombo(), { wrapper });
    expect(result.current.active).toEqual({
      instrument: 'XAUUSD',
      timeframe: 'M15',
    });
  });

  it('throws when used outside the provider', () => {
    expect(() => renderHook(() => useActiveCombo())).toThrow(
      /ActiveComboProvider/,
    );
  });
});
