'use client';

import * as React from 'react';

/**
 * A multi-select filter group over a fixed set of factual values (state, type,
 * side, TF — never a rank/score). Shared by the Structure and Liquidity cards so
 * the selection logic lives in ONE place.
 *
 * Contract (mission UI-3a):
 *   · each value is an independent toggle; several can be active at once;
 *   · the default is EVERYTHING selected (⇒ show all, the prior behaviour);
 *   · `reset()` re-selects the whole group — it is an ACTION, not a "Tous" state;
 *   · when NOTHING is selected (`noneSelected`) the caller shows zero results and
 *     an explicit message — it must NEVER silently fall back to "show all".
 */
export interface MultiFilter<T extends string> {
  selected: ReadonlySet<T>;
  isOn(value: T): boolean;
  toggle(value: T): void;
  reset(): void;
  /** True when the group has no value selected → the caller renders zero rows. */
  noneSelected: boolean;
}

export function useMultiFilter<T extends string>(all: readonly T[]): MultiFilter<T> {
  const [selected, setSelected] = React.useState<Set<T>>(() => new Set(all));

  const toggle = React.useCallback((value: T) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(value)) next.delete(value);
      else next.add(value);
      return next;
    });
  }, []);

  const reset = React.useCallback(() => setSelected(new Set(all)), [all]);

  const isOn = React.useCallback((value: T) => selected.has(value), [selected]);

  return { selected, isOn, toggle, reset, noneSelected: selected.size === 0 };
}
