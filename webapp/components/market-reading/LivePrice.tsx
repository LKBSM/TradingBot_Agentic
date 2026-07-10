'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import { cn } from '@/lib/utils';

interface LivePriceProps {
  instrument: string;
  /** Unified last price (M15 freshest close), already resolved upstream. */
  price: number;
  /** Fractional daily change (e.g. -0.0322), or null when no reference yet. */
  changePct: number | null;
}

const TONE_CLASS: Record<'bull' | 'bear' | 'neutral' | 'warn', string> = {
  bull: 'text-emerald-600 dark:text-emerald-400',
  bear: 'text-red-600 dark:text-red-400',
  neutral: 'text-muted-foreground',
  warn: 'text-amber-600 dark:text-amber-400',
};

/**
 * Header price + descriptive daily change. The price is unified across
 * timeframes (see useLatestPrice / computeDailyChange) and flashes discreetly
 * when it changes — a subtle "alive" cue, NOT a tick stream (the value only
 * refreshes on a coarse interval / candle close upstream).
 */
export function LivePrice({ instrument, price, changePct }: LivePriceProps) {
  const t = useTranslations('reading.temporal');
  const fmt = useReadingFormatters();
  const [flash, setFlash] = React.useState(false);
  const prev = React.useRef(price);

  React.useEffect(() => {
    if (prev.current !== price) {
      prev.current = price;
      setFlash(true);
      const id = window.setTimeout(() => setFlash(false), 650);
      return () => window.clearTimeout(id);
    }
    return undefined;
  }, [price]);

  const tone = fmt.changeTone(changePct);

  return (
    <span className="flex flex-wrap items-baseline justify-end gap-x-2 gap-y-0.5">
      <span
        className={cn(
          'font-mono text-lg font-medium tabular-nums transition-colors duration-700',
          flash ? TONE_CLASS[tone] : 'text-foreground',
        )}
        aria-live="polite"
      >
        {fmt.price(price, instrument)}
      </span>
      {changePct !== null && (
        <span
          className={cn('font-mono text-xs font-medium tabular-nums', TONE_CLASS[tone])}
          aria-label={t('dailyChangeAria', { pct: fmt.changePercent(changePct) })}
        >
          {fmt.changePercent(changePct)}
        </span>
      )}
    </span>
  );
}
