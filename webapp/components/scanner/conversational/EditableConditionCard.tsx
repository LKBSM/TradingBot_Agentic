'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import type { ControlName, ScanCondition } from '@/lib/conditions/types';
import { paletteEntry } from '@/lib/conditions/palette';
import { cn } from '@/lib/utils';
import { useConditionLabels } from './labels';

/**
 * One translated condition rendered as an EDITABLE card (state 2/3): family tag,
 * the EXACT palette label, its segmented value controls, and a remove button.
 * Segmented controls show every option at once (never a dropdown) — the value
 * the user picks is always inside the palette domain, so editing can never
 * produce an out-of-palette condition.
 */
export function EditableConditionCard({
  condition,
  index,
  assumedControls,
  onChange,
  onRemove,
}: {
  condition: ScanCondition;
  index: number;
  /** Control names whose value M.I.A assumed — flagged inline for honesty. */
  assumedControls?: ReadonlySet<ControlName>;
  onChange(index: number, next: ScanCondition): void;
  onRemove(index: number): void;
}) {
  const t = useTranslations('scannerChat');
  const { conditionLabel, optionLabel, familyLabel } = useConditionLabels();
  const entry = paletteEntry(condition.type);
  if (!entry) return null;

  return (
    <div
      data-testid="translated-card"
      data-type={condition.type}
      className="relative rounded-xl border border-primary/40 bg-card/60 p-3"
    >
      <button
        type="button"
        aria-label={t('card.remove')}
        title={t('card.remove')}
        onClick={() => onRemove(index)}
        className="absolute right-2 top-2 grid h-6 w-6 place-items-center rounded-md border border-border/60 text-muted-foreground hover:border-destructive hover:text-destructive"
      >
        ×
      </button>
      <div className="mb-1 font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
        {familyLabel(entry.family)}
      </div>
      <div className="mb-2 pr-6 text-sm font-medium text-foreground">
        {conditionLabel(condition.type)}
      </div>

      <div className="space-y-2">
        {entry.controls.map((control) => {
          const name = control.name as ControlName;
          const current = (condition[name] as string | number | undefined) ?? control.default;
          const assumed = assumedControls?.has(name) ?? false;
          return (
            <div key={name}>
              <div className="flex flex-wrap gap-1">
                {control.values.map((value) => {
                  const active = String(current) === String(value);
                  return (
                    <button
                      key={String(value)}
                      type="button"
                      aria-pressed={active}
                      onClick={() => onChange(index, { ...condition, [name]: value })}
                      className={cn(
                        'rounded-md border px-2 py-1 font-mono text-[10px] transition',
                        active
                          ? 'border-primary/60 bg-primary/15 text-primary'
                          : 'border-border/60 bg-background/60 text-muted-foreground hover:border-border',
                      )}
                    >
                      {optionLabel(name, value)}
                    </button>
                  );
                })}
              </div>
              {assumed && (
                <p className="mt-1 font-mono text-[9px] text-amber-600">{t('card.assumedInline')}</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
