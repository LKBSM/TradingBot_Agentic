'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';

/**
 * Single-choice segmented control (SC-1): ALL options are visible at once as
 * buttons — never a dropdown. When ``disabled`` the whole group is dimmed and
 * inert (a decoupled condition's values are shown but not interactive).
 */
export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  disabled = false,
  ariaLabel,
}: {
  options: Array<{ value: T; label: string }>;
  value: T;
  onChange(value: T): void;
  disabled?: boolean;
  ariaLabel?: string;
}) {
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      className={cn(
        'inline-flex flex-wrap gap-1 rounded-md border border-border/60 bg-background/40 p-0.5',
        disabled && 'pointer-events-none opacity-40',
      )}
    >
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={String(opt.value)}
            type="button"
            aria-pressed={active}
            disabled={disabled}
            onClick={() => onChange(opt.value)}
            className={cn(
              'rounded px-2 py-1 text-xs font-medium transition-colors',
              active
                ? 'bg-foreground text-background'
                : 'text-muted-foreground hover:text-foreground',
            )}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
