'use client';

import { useLocale, useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import { formatZoneDateTime, type TimelineEvent } from '@/lib/zones/lifecycle';

/**
 * Vertical lifecycle timeline for one zone. Renders ONLY the events handed to it
 * (`buildTimeline` never fabricates a step), and shows a date only when the
 * engine recorded one — an event without a timestamp degrades to its label
 * alone. Purely descriptive: a sequence of facts already produced upstream.
 */

const DOT_TONE: Record<TimelineEvent['variant'], string> = {
  formed: 'bg-sky-500',
  interaction: 'bg-amber-500',
  terminal: 'bg-rose-500',
  ongoing: 'bg-emerald-500',
};

export function ZoneTimeline({ events }: { events: TimelineEvent[] }) {
  const t = useTranslations('zones');
  const locale = useLocale();
  return (
    <ol className="relative ml-1 flex flex-col gap-3 border-l border-border/70 pl-4">
      {events.map((ev, i) => (
        <li key={`${ev.key}-${i}`} className="relative">
          <span
            aria-hidden
            className={cn(
              'absolute -left-[1.30rem] top-1 h-2.5 w-2.5 rounded-full ring-2 ring-background',
              DOT_TONE[ev.variant],
              ev.variant === 'ongoing' && 'animate-pulse',
            )}
          />
          <div className="flex flex-col">
            <span className="text-sm font-medium text-foreground">{ev.label}</span>
            <span className="text-xs text-muted-foreground">
              {ev.at
                ? formatZoneDateTime(ev.at, locale)
                : ev.variant === 'ongoing'
                  ? t('timeline.now')
                  : t('timeline.noDate')}
              {/* The engine records only the FIRST interaction (no per-test
                  history) — say so rather than let the date read as "the" test. */}
              {ev.variant === 'interaction' && ev.at ? ` · ${t('timeline.firstContact')}` : ''}
            </span>
          </div>
        </li>
      ))}
    </ol>
  );
}
