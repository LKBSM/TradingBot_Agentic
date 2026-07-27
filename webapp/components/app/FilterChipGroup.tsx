'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';
import type { MultiFilter } from '@/lib/market-reading/use-multi-filter';

/**
 * A labelled row of multi-select filter chips + a « Réinitialiser » action
 * (mission UI-3a). Shared by the Structure and Liquidity cards — same tokens and
 * dimensions as the existing `.fchip`; only the behaviour changes.
 *
 * - Each chip is a real <button> with `aria-pressed` reflecting its state and is
 *   keyboard-navigable. The active state is signalled by a check glyph (CSS
 *   `.fchip.on::before`), not colour alone.
 * - « Réinitialiser » is an ACTION (re-selects the whole group), NOT a toggle —
 *   it carries no `aria-pressed`. It replaces the removed "Tous"/"Les deux" chip.
 */
export function FilterChipGroup<T extends string>({
  label,
  options,
  filter,
  resetLabel,
}: {
  label: string;
  options: readonly { value: T; label: string }[];
  filter: MultiFilter<T>;
  resetLabel: string;
}) {
  return (
    <>
      <div className="fsec">{label}</div>
      <div className="fgrp">
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            className={cn('fchip', filter.isOn(o.value) && 'on')}
            aria-pressed={filter.isOn(o.value)}
            onClick={() => filter.toggle(o.value)}
          >
            {o.label}
          </button>
        ))}
        <button type="button" className="fchip freset" onClick={filter.reset}>
          {resetLabel}
        </button>
      </div>
    </>
  );
}
