'use client';

import * as React from 'react';
import { SUPPORTED_INSTRUMENTS, SUPPORTED_TIMEFRAMES } from './perimeter';

/**
 * Lightweight shared state for the active (instrument, timeframe) combination
 * selected in the /app view. A plain React Context — no external state library;
 * the surface is intentionally tiny (one selected combo + a setter).
 */

export interface Combo {
  instrument: string;
  timeframe: string;
}

/**
 * V1 perimeter per the backend endpoint (SUPPORTED_INSTRUMENTS ×
 * SUPPORTED_TIMEFRAMES = XAUUSD/EURUSD × M15/H1/H4). Order is the display
 * order in the instruments column.
 */
export { SUPPORTED_INSTRUMENTS, SUPPORTED_TIMEFRAMES };

export const SUPPORTED_COMBOS: readonly Combo[] = SUPPORTED_INSTRUMENTS.flatMap(
  (instrument) =>
    SUPPORTED_TIMEFRAMES.map((timeframe) => ({ instrument, timeframe })),
);

/** Stable string key for a combo (used for React keys + equality). */
export function comboKey(combo: Combo): string {
  return `${combo.instrument}:${combo.timeframe}`;
}

export function sameCombo(a: Combo | null, b: Combo | null): boolean {
  if (a === null || b === null) return a === b;
  return a.instrument === b.instrument && a.timeframe === b.timeframe;
}

interface ActiveComboContextValue {
  /** Currently selected combo, or null when nothing is selected yet. */
  active: Combo | null;
  /** Select a combo (null clears the selection). */
  select(combo: Combo | null): void;
  /** The full V1 perimeter, for rendering the instruments column. */
  combos: readonly Combo[];
}

const ActiveComboContext = React.createContext<ActiveComboContextValue | null>(
  null,
);

export function ActiveComboProvider({
  children,
  initial = null,
}: {
  children: React.ReactNode;
  initial?: Combo | null;
}) {
  const [active, setActive] = React.useState<Combo | null>(initial);

  const select = React.useCallback((combo: Combo | null) => {
    setActive(combo);
  }, []);

  const value = React.useMemo<ActiveComboContextValue>(
    () => ({ active, select, combos: SUPPORTED_COMBOS }),
    [active, select],
  );

  return (
    <ActiveComboContext.Provider value={value}>
      {children}
    </ActiveComboContext.Provider>
  );
}

export function useActiveCombo(): ActiveComboContextValue {
  const ctx = React.useContext(ActiveComboContext);
  if (!ctx) {
    throw new Error(
      'useActiveCombo must be used inside an <ActiveComboProvider />.',
    );
  }
  return ctx;
}
