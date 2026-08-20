'use client';

import { Clock } from 'lucide-react';
import * as React from 'react';
import { useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import { formatLocalHm } from '@/lib/time/localTime';

interface PriceFreshnessBadgeProps {
  /**
   * Epoch SECONDS of the candle/tick that produced the displayed REFERENCE price
   * (e.g. `priceTs` from useLatestPrice/computeDailyChange). Null when unknown —
   * the badge then renders nothing (honest: no freshness to claim).
   */
  tsSec: number | null | undefined;
  className?: string;
}

/**
 * Surfaces the freshness of the REFERENCE price a card shows — the exact local
 * time it was read plus its relative age (« Prix à 14:32 · il y a 12 min »).
 *
 * Why (mission diag/price-freshness-zone-card): the reference price feeding the
 * zone proximity / the « Prix courant » row is the last CLOSED candle, refreshed
 * on a coarse cadence — while the chart's price line can follow a fresher (live)
 * source. Without a timestamp, a legitimate few-seconds lag is indistinguishable
 * from a real staleness problem. The epoch already exists internally (`priceTs`);
 * this only makes it legible. No new data, no extra request.
 *
 * Re-renders every 30 s to keep the age current. Renders the exact time only
 * before mount (Date/timezone are not deterministic during SSR), then fills in
 * the relative age — same guard as TemporalBadge, no hydration mismatch.
 */
export function PriceFreshnessBadge({ tsSec, className }: PriceFreshnessBadgeProps) {
  const t = useTranslations('reading.temporal');
  const fmt = useReadingFormatters();
  const [now, setNow] = React.useState<Date | null>(null);

  React.useEffect(() => {
    setNow(new Date());
    const id = window.setInterval(() => setNow(new Date()), 30_000);
    return () => window.clearInterval(id);
  }, []);

  if (tsSec == null || !Number.isFinite(tsSec)) return null;

  const at = new Date(tsSec * 1000);
  const time = formatLocalHm(at);
  const label =
    now === null
      ? t('priceAt', { time })
      : t('priceFreshness', { time, rel: fmt.relativePast(at.toISOString(), now) });

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 font-mono text-[11px] font-normal text-muted-foreground',
        className,
      )}
      title={t('priceAtTitle')}
      role="status"
      data-testid="price-freshness"
    >
      <Clock className="h-3.5 w-3.5" aria-hidden />
      {label}
    </span>
  );
}
