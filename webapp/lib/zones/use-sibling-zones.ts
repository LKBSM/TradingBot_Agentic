'use client';

import * as React from 'react';
import { fetchMarketReading } from '@/lib/market-reading/api-client';
import { SUPPORTED_TIMEFRAMES } from '@/lib/market-reading/perimeter';
import { getMockReading, READING_DATA_SOURCE } from '@/lib/mockReadings';
import type { ReadingSource } from '@/lib/market-reading/hooks';
import { collectZones, type SiblingZone } from './lifecycle';

/**
 * Zones detected on the OTHER supported timeframes of the same instrument, for
 * the geometric-overlap fact on the /zones cards ("chevauche un OB H1 (bornes)").
 *
 * Read-only over the SAME engine readings the /app surface consumes — one
 * cache-served `fetchMarketReading` per sibling timeframe (the pattern
 * `useMtfTrends` already uses), zero new detection. A failed / unavailable
 * timeframe contributes NOTHING (the overlap line is simply absent for it) —
 * honest degradation, never an inferred zone.
 */
export interface UseSiblingZonesResult {
  siblings: SiblingZone[];
  isLoading: boolean;
}

function toSiblings(
  structure: Parameters<typeof collectZones>[0],
  timeframe: string,
): SiblingZone[] {
  return collectZones(structure).map((z) => ({
    id: z.id,
    kind: z.kind,
    direction: z.direction,
    levelHigh: z.levelHigh,
    levelLow: z.levelLow,
    timeframe,
  }));
}

export function useSiblingZones(
  instrument: string | null,
  timeframe: string | null,
  options: { source?: ReadingSource } = {},
): UseSiblingZonesResult {
  const { source = READING_DATA_SOURCE } = options;
  const [siblings, setSiblings] = React.useState<SiblingZone[]>([]);
  const [isLoading, setIsLoading] = React.useState(false);
  const seqRef = React.useRef(0);

  React.useEffect(() => {
    if (!instrument || !timeframe) {
      setSiblings([]);
      setIsLoading(false);
      return;
    }

    const others = SUPPORTED_TIMEFRAMES.filter((tf) => tf !== timeframe);
    const seq = ++seqRef.current;
    setSiblings([]);
    setIsLoading(true);

    // ── Mock source: resolve locally, no network. ──
    if (source === 'mock') {
      const out: SiblingZone[] = [];
      for (const tf of others) {
        const reading = getMockReading(instrument, tf);
        if (reading) out.push(...toSiblings(reading.structure, tf));
      }
      if (seq === seqRef.current) {
        setSiblings(out);
        setIsLoading(false);
      }
      return;
    }

    const controller = new AbortController();
    Promise.all(
      others.map((tf) =>
        fetchMarketReading(instrument, tf, { signal: controller.signal })
          .then((r) => toSiblings(r.structure, tf))
          // Unavailable sibling → no overlap facts for it (never inferred).
          .catch(() => [] as SiblingZone[]),
      ),
    ).then((lists) => {
      if (seq !== seqRef.current) return;
      setSiblings(lists.flat());
      setIsLoading(false);
    });

    return () => controller.abort();
  }, [instrument, timeframe, source]);

  return { siblings, isLoading };
}
