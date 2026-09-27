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
import { ALL_MARKET_IDS } from '@/lib/markets';
import { DISPLAY_TIMEFRAMES } from '@/lib/market-reading/perimeter';

describe('active-combo store', () => {
  it('exposes every market × the five displayed units (M1 gated off)', () => {
    // DATA-4 : la liste n'est plus écrite à la main — 80 marchés × 5 unités =
    // 400 combinaisons. Ce que ce test garde, c'est la RÈGLE qui les engendre :
    // chaque marché du registre croisé avec les 5 unités affichées, dans cet
    // ordre, et M1 jamais offerte.
    expect(DISPLAY_TIMEFRAMES).not.toContain('M1');
    expect(SUPPORTED_COMBOS).toHaveLength(ALL_MARKET_IDS.length * DISPLAY_TIMEFRAMES.length);

    const expected = ALL_MARKET_IDS.flatMap((instrument) =>
      DISPLAY_TIMEFRAMES.map((timeframe) => `${instrument}:${timeframe}`),
    );
    expect(SUPPORTED_COMBOS.map(comboKey)).toEqual(expected);

    // Les deux marchés d'origine restent servis, avec leurs cinq unités.
    expect(SUPPORTED_COMBOS.map(comboKey)).toEqual(
      expect.arrayContaining(['XAUUSD:M5', 'XAUUSD:D1', 'EURUSD:M5', 'EURUSD:D1']),
    );
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
    expect(result.current.combos).toHaveLength(SUPPORTED_COMBOS.length);

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
