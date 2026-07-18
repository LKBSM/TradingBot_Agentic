'use client';

import { Clock } from 'lucide-react';
import * as React from 'react';
import { useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';

interface TemporalBadgeProps {
  /** ISO-8601 timestamp of the candle close this reading describes. */
  candleCloseTs: string;
  className?: string;
}

/**
 * Shows when the candle this reading describes closed ("Bougie clôturée il y a
 * X"). Re-renders every 30 s to keep the relative age fresh without burning
 * CPU. Renders neutrally during SSR (Date is not deterministic) and fills in
 * the real value after mount.
 */
export function TemporalBadge({ candleCloseTs, className }: TemporalBadgeProps) {
  const t = useTranslations('reading.temporal');
  const fmt = useReadingFormatters();
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
          'flex items-center gap-2 font-mono text-[11px] font-normal text-muted-foreground',
          className,
        )}
        aria-hidden
      >
        <Clock className="h-3.5 w-3.5" />
        <span className="opacity-0">Chargement…</span>
      </div>
    );
  }

  return (
    <div
      className={cn(
        'flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] font-normal text-muted-foreground',
        className,
      )}
    >
      <span className="inline-flex items-center gap-1.5">
        <Clock className="h-3.5 w-3.5" aria-hidden />
        {t('candleClosed', { rel: fmt.relativePast(candleCloseTs, now) })}
      </span>
    </div>
  );
}
