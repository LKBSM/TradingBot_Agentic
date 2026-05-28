'use client';

import { Clock, AlertTriangle } from 'lucide-react';
import * as React from 'react';
import { cn } from '@/lib/utils';
import {
  formatRelativePast,
  formatValidityCountdown,
} from '@/lib/insight-formatters';

interface TemporalBadgeProps {
  createdAtUtc: string;
  validUntilUtc: string;
  className?: string;
}

/**
 * Shows "il y a X" relative time + a live-ticking countdown to validity
 * expiration. Re-renders every 30 s to keep the countdown fresh without
 * burning CPU. Renders neutrally during SSR (Date is not deterministic) and
 * fills in real values after mount.
 */
export function TemporalBadge({
  createdAtUtc,
  validUntilUtc,
  className,
}: TemporalBadgeProps) {
  const [now, setNow] = React.useState<Date | null>(null);

  React.useEffect(() => {
    setNow(new Date());
    const id = window.setInterval(() => setNow(new Date()), 30_000);
    return () => window.clearInterval(id);
  }, []);

  if (now === null) {
    return (
      <div
        className={cn(
          'flex items-center gap-2 text-xs text-muted-foreground',
          className,
        )}
        aria-hidden
      >
        <Clock className="h-3.5 w-3.5" />
        <span className="opacity-0">Chargement…</span>
      </div>
    );
  }

  const past = formatRelativePast(createdAtUtc, now);
  const validity = formatValidityCountdown(validUntilUtc, now);

  return (
    <div
      className={cn(
        'flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground',
        className,
      )}
    >
      <span className="inline-flex items-center gap-1.5">
        <Clock className="h-3.5 w-3.5" aria-hidden />
        Émise {past}
      </span>
      <span
        className={cn(
          'inline-flex items-center gap-1.5',
          validity.expired && 'text-sentinel-warn',
        )}
      >
        {validity.expired && (
          <AlertTriangle className="h-3.5 w-3.5" aria-hidden />
        )}
        Lecture {validity.label}
      </span>
    </div>
  );
}
