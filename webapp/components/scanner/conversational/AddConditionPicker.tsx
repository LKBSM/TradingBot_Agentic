'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import type { ConditionType, ScanCondition } from '@/lib/conditions/types';
import { CONDITION_PALETTE, groupByFamily } from '@/lib/conditions/palette';
import { cn } from '@/lib/utils';
import { useConditionLabels } from './labels';

/**
 * "Ajouter depuis la palette complète" (state 2). The conversational mode is an
 * ADDITIONAL entry, never a replacement: the full closed palette stays one click
 * away so the user can compose by hand. Adding a condition seeds it with the
 * palette defaults — never a value inferred from nothing.
 */
export function AddConditionPicker({
  present,
  onAdd,
}: {
  present: ReadonlySet<ConditionType>;
  onAdd(condition: ScanCondition): void;
}) {
  const t = useTranslations('scannerChat');
  const { conditionLabel, familyLabel } = useConditionLabels();
  const [open, setOpen] = React.useState(false);
  const groups = React.useMemo(() => groupByFamily(CONDITION_PALETTE), []);

  if (!open) {
    return (
      <button
        type="button"
        data-testid="open-palette"
        onClick={() => setOpen(true)}
        className="flex h-full min-h-[92px] w-full flex-col items-start justify-center rounded-xl border border-dashed border-border/70 bg-background/40 p-3 text-left text-muted-foreground hover:border-primary/50 hover:text-foreground"
      >
        <span className="font-mono text-[9px] uppercase tracking-wider">{t('add.tag')}</span>
        <span className="text-sm font-medium">{t('add.cta')}</span>
        <span className="font-mono text-[10px]">{t('add.hint')}</span>
      </button>
    );
  }

  return (
    <div className="rounded-xl border border-border/70 bg-card/60 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-medium text-foreground">{t('add.title')}</span>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="rounded-md border border-border/60 px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
        >
          {t('add.close')}
        </button>
      </div>
      <div className="space-y-3">
        {groups.map(({ family, entries }) => (
          <div key={family}>
            <div className="mb-1 font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
              {familyLabel(family)}
            </div>
            <div className="flex flex-wrap gap-1">
              {entries.map((entry) => {
                const already = present.has(entry.type);
                return (
                  <button
                    key={entry.type}
                    type="button"
                    disabled={already}
                    data-testid="palette-option"
                    onClick={() => {
                      onAdd(seedCondition(entry.type));
                      setOpen(false);
                    }}
                    className={cn(
                      'rounded-md border px-2 py-1 text-left text-xs transition',
                      already
                        ? 'cursor-not-allowed border-border/40 text-muted-foreground/50'
                        : 'border-border/60 text-muted-foreground hover:border-primary/50 hover:text-foreground',
                    )}
                    title={already ? t('add.already') : undefined}
                  >
                    {conditionLabel(entry.type)}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/** A new condition with only its palette defaults — no inferred values. */
function seedCondition(type: ConditionType): ScanCondition {
  return { type };
}
