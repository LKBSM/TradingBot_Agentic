'use client';

import * as React from 'react';
import { fetchCalendar } from './api';
import type { CalendarResponse } from '@/types/calendar';

export interface UseCalendarResult {
  data: CalendarResponse | null;
  isLoading: boolean;
  error: Error | null;
  refresh(): void;
}

/**
 * Fetch the calendar window once (+ optional poll). Light state, mirroring the
 * market-reading hooks: no SWR/React-Query. Stale responses are discarded.
 */
export function useCalendar(options: {
  lookaheadDays?: number;
  lookbackDays?: number;
  pollMs?: number;
} = {}): UseCalendarResult {
  const { lookaheadDays = 7, lookbackDays = 3, pollMs } = options;
  const [data, setData] = React.useState<CalendarResponse | null>(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState<Error | null>(null);
  const [nonce, setNonce] = React.useState(0);
  const seq = React.useRef(0);

  const refresh = React.useCallback(() => setNonce((n) => n + 1), []);

  React.useEffect(() => {
    const token = ++seq.current;
    const controller = new AbortController();
    let cancelled = false;

    fetchCalendar({ lookaheadDays, lookbackDays, signal: controller.signal })
      .then((resp) => {
        if (cancelled || token !== seq.current) return;
        setData(resp);
        setError(null);
      })
      .catch((err: unknown) => {
        if (cancelled || token !== seq.current) return;
        setError(err instanceof Error ? err : new Error('calendar error'));
      })
      .finally(() => {
        if (cancelled || token !== seq.current) return;
        setIsLoading(false);
      });

    let timer: ReturnType<typeof setInterval> | undefined;
    if (pollMs && pollMs > 0) {
      timer = setInterval(refresh, pollMs);
    }
    return () => {
      cancelled = true;
      controller.abort();
      if (timer) clearInterval(timer);
    };
  }, [lookaheadDays, lookbackDays, pollMs, nonce, refresh]);

  return { data, isLoading, error, refresh };
}
