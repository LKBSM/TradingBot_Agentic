'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';

/**
 * A small accessible switch for the Scanner's auto-refresh preference.
 * Neutral, descriptive wording — it only controls WHEN we re-read, never what
 * the scanner claims. Aligned on candle closes; this just turns that on/off.
 */
export function AutoRefreshToggle({
  enabled,
  onChange,
}: {
  enabled: boolean;
  onChange(next: boolean): void;
}) {
  const t = useTranslations('scanner.toggle');
  return (
    <label className="flex cursor-pointer select-none items-center gap-2">
      <span>{t('label')}</span>
      <button
        type="button"
        role="switch"
        aria-checked={enabled}
        aria-label={t('aria')}
        onClick={() => onChange(!enabled)}
        className={cn(
          // Visual pill stays 16×28; the ::after extends the tap area to 44×44
          // on touch without shifting layout (RESP-E-07).
          'relative inline-flex h-4 w-7 shrink-0 items-center rounded-full transition-colors',
          "after:absolute after:-inset-x-2 after:-inset-y-3.5 after:content-['']",
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1',
          enabled ? 'bg-primary' : 'bg-muted-foreground/30',
        )}
      >
        <span
          className={cn(
            'inline-block h-3 w-3 transform rounded-full bg-background shadow transition-transform',
            enabled ? 'translate-x-3.5' : 'translate-x-0.5',
          )}
        />
      </button>
    </label>
  );
}
