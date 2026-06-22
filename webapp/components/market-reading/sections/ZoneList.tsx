'use client';

import { useMemo, useState } from 'react';
import { cn } from '@/lib/utils';

/**
 * ZoneList — collapsible, ranked rendering of SMC zones (Order Blocks / Fair
 * Value Gaps) for the « Structure de marché » panel.
 *
 * Display / quality only — this NEVER touches detection. It re-orders and folds
 * the zones the engine already emitted:
 *   1. identical zones are de-duplicated (same bounds + status + importance),
 *   2. the remainder is ranked by importance, then by proximity to the current
 *      price (neutral — a contrary zone is folded, never hidden/removed),
 *   3. only the top `collapsedCount` (default 3) show by default; a « voir plus »
 *      toggle reveals the FULL list (« voir moins » folds it back).
 *
 * Everything stays accessible: nothing is dropped, the count is surfaced.
 */
export interface ZoneListProps<T> {
  zones: T[];
  /** Current price — drives the proximity tie-break. Optional: when absent the
   *  list ranks by importance only (proximity contributes 0). */
  price?: number;
  /** Lower rank = more important (e.g. high=0, medium=1, low=2). */
  importanceRank: (zone: T) => number;
  /** [low, high] band of the zone, for the proximity tie-break. */
  band: (zone: T) => [number, number];
  /** The fr-FR descriptive line for the zone (band · importance · status …). */
  renderLabel: (zone: T) => string;
  /** Whether the zone is still active (vs mitigated / filled) — drives the
   *  active/mitigated visual distinction. */
  isActive: (zone: T) => boolean;
  /** Stable identity key — identical keys are folded as one zone. */
  dedupKey: (zone: T) => string;
  /** How many zones to show before « voir plus ». */
  collapsedCount?: number;
  /** Accessible noun for the toggle, e.g. "zone". */
  noun: string;
}

function bandDistance(low: number, high: number, price: number): number {
  if (price >= low && price <= high) return 0;
  return price < low ? low - price : price - high;
}

export function ZoneList<T>({
  zones,
  price,
  importanceRank,
  band,
  renderLabel,
  isActive,
  dedupKey,
  collapsedCount = 3,
  noun,
}: ZoneListProps<T>) {
  const [expanded, setExpanded] = useState(false);

  const ordered = useMemo(() => {
    // 1. De-duplicate identical zones (keep first occurrence).
    const seen = new Set<string>();
    const unique: T[] = [];
    for (const z of zones) {
      const key = dedupKey(z);
      if (seen.has(key)) continue;
      seen.add(key);
      unique.push(z);
    }
    // 2. Rank by importance, then proximity to the current price.
    return unique
      .map((zone) => {
        const [low, high] = band(zone);
        const distance =
          price === undefined ? 0 : bandDistance(low, high, price);
        return { zone, distance };
      })
      .sort(
        (a, b) =>
          importanceRank(a.zone) - importanceRank(b.zone) ||
          a.distance - b.distance,
      )
      .map((entry) => entry.zone);
  }, [zones, price, importanceRank, band, dedupKey]);

  const hidden = Math.max(0, ordered.length - collapsedCount);
  const visible = expanded ? ordered : ordered.slice(0, collapsedCount);

  return (
    <div className="space-y-2">
      <ul className="space-y-1.5">
        {visible.map((zone, i) => {
          const active = isActive(zone);
          return (
            <li
              key={dedupKey(zone) + i}
              className={cn(
                'flex items-start gap-2 text-sm',
                active ? 'text-foreground' : 'text-muted-foreground',
              )}
            >
              <span
                aria-hidden
                className={cn(
                  'mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full',
                  active
                    ? 'bg-foreground/70'
                    : 'border border-muted-foreground/50 bg-transparent',
                )}
              />
              <span className="font-medium">{renderLabel(zone)}</span>
            </li>
          );
        })}
      </ul>

      {hidden > 0 && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          className="text-xs font-medium text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
        >
          {expanded
            ? '▴ voir moins'
            : `▾ voir plus (${hidden} ${noun}${hidden > 1 ? 's' : ''})`}
        </button>
      )}
    </div>
  );
}
