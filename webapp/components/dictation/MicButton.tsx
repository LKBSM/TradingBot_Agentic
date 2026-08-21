'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';

/**
 * The single microphone affordance shared by every M.I.A chat surface. Three
 * visual states reflect the REAL dictation state:
 *   · idle      — neutral outline, invites a click;
 *   · listening — primary fill + ring (aria-pressed), click stops;
 *   · denied    — amber alert outline so a refused permission is self-evident;
 *                 clicking retries (the browser may re-prompt), and the field
 *                 stays fully usable at the keyboard either way.
 *
 * The button is only ever rendered when dictation is feature-supported, so there
 * is never a dead button (see each caller's `voice.supported` guard).
 */
export function MicButton({
  listening,
  denied,
  onToggle,
  startLabel,
  stopLabel,
  className,
}: {
  listening: boolean;
  denied: boolean;
  /** Start when idle, stop when listening — the caller's toggle handles both. */
  onToggle(): void;
  startLabel: string;
  stopLabel: string;
  /** Size / spacing override; defaults to the 36px square used on the scanner. */
  className?: string;
}) {
  const label = listening ? stopLabel : startLabel;
  return (
    <button
      type="button"
      data-testid="mic-button"
      data-state={listening ? 'listening' : denied ? 'denied' : 'idle'}
      aria-pressed={listening}
      aria-label={label}
      title={label}
      onClick={onToggle}
      className={cn(
        'grid h-9 w-9 shrink-0 place-items-center rounded-lg border transition',
        listening
          ? 'border-primary bg-primary/15 text-primary'
          : denied
            ? 'border-amber-500/70 bg-amber-500/10 text-amber-600 hover:border-amber-500 dark:text-amber-500'
            : 'border-border/70 text-muted-foreground hover:border-primary hover:text-primary',
        className,
      )}
    >
      <MicIcon />
    </button>
  );
}

export function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.9" aria-hidden>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0014 0M12 18v3" strokeLinecap="round" />
    </svg>
  );
}
