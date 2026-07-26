'use client';

import { useLocale, useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import { formatZoneShortTime, type TimelineEvent } from '@/lib/zones/lifecycle';

/**
 * Horizontal lifecycle timeline for one zone (UI-2 terminal reference: `.tl` /
 * `.step`). Renders ONLY the events handed to it (`buildTimeline` never
 * fabricates a step) and shows a time only when the engine recorded one — an
 * event without a timestamp degrades to a dash. Purely descriptive: a sequence
 * of facts already produced upstream.
 *
 * Honesty: the interaction step is the single « Testé / Pénétré » — the engine
 * records only the FIRST contact (no per-test history), so there is NEVER a
 * « ×N » and no per-test list.
 */

// Each engine event variant maps onto a reference step state:
//   · ongoing (live) → `.now`  (the amber-highlighted "Maintenant" node)
//   · everything else (formed / interaction / terminal) → `.done`
function stepClass(variant: TimelineEvent['variant']): string {
  return variant === 'ongoing' ? 'now' : 'done';
}

export function ZoneTimeline({
  events,
  /** Sub-label shown under the live "Maintenant" step (e.g. « prix dedans »). */
  nowSubLabel,
}: {
  events: TimelineEvent[];
  nowSubLabel?: string;
}) {
  const t = useTranslations('zones');
  const locale = useLocale();
  return (
    <div className="tl">
      {events.map((ev, i) => {
        // `.st3` — the small mono sub-line: the recorded time when there is one;
        // for the live step the caller's factual sub-label (or "à présent");
        // otherwise a neutral dash (never a fabricated date).
        const sub = ev.at
          ? formatZoneShortTime(ev.at, locale)
          : ev.variant === 'ongoing'
            ? (nowSubLabel ?? t('timeline.now'))
            : t('timeline.noDate');
        return (
          <div key={`${ev.key}-${i}`} className={cn('step', stepClass(ev.variant))}>
            <div className="d3" aria-hidden />
            <div className="sl">{ev.label}</div>
            <div className="st3">{sub}</div>
          </div>
        );
      })}
    </div>
  );
}
