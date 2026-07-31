'use client';

import * as React from 'react';
import {
  fetchCalendar,
  fetchCalendarEvent,
  fetchCalendarMonth,
  fetchPublicationMeasures,
} from './api';
import type { CalendarResponse } from '@/types/calendar';
import type { PublicationMeasures } from '@/types/measures';

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

/**
 * Fetch every attached event within one calendar month ('YYYY-MM'). Backs the
 * month grid; re-fetches whenever the target month changes.
 */
export function useCalendarMonth(month: string): UseCalendarResult {
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
    setIsLoading(true);

    fetchCalendarMonth(month, { signal: controller.signal })
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

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [month, nonce]);

  return { data, isLoading, error, refresh };
}

export interface UseMeasuresResult {
  data: PublicationMeasures | null;
  isLoading: boolean;
  error: Error | null;
}

/**
 * Fetch the engine-measured facts for a recurring publication (by event key).
 * Empty key ⇒ no fetch (null data), so callers can guard on measurability.
 */
export function usePublicationMeasures(eventKey: string | null): UseMeasuresResult {
  const [data, setData] = React.useState<PublicationMeasures | null>(null);
  const [isLoading, setIsLoading] = React.useState(!!eventKey);
  const [error, setError] = React.useState<Error | null>(null);
  const seq = React.useRef(0);

  React.useEffect(() => {
    if (!eventKey) {
      setData(null);
      setIsLoading(false);
      setError(null);
      return;
    }
    const token = ++seq.current;
    const controller = new AbortController();
    let cancelled = false;
    setIsLoading(true);

    fetchPublicationMeasures(eventKey, { signal: controller.signal })
      .then((resp) => {
        if (cancelled || token !== seq.current) return;
        setData(resp);
        setError(null);
      })
      .catch((err: unknown) => {
        if (cancelled || token !== seq.current) return;
        setError(err instanceof Error ? err : new Error('measures error'));
      })
      .finally(() => {
        if (cancelled || token !== seq.current) return;
        setIsLoading(false);
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [eventKey]);

  return { data, isLoading, error };
}

/**
 * Fetch ONE event by its stable id (REC point 1). The per-event detail must not
 * depend on a list window — a deep-linked event exists by definition. Returns
 * the same CalendarResponse shape, holding the single event (or empty when the
 * id genuinely does not exist).
 */
export function useCalendarEvent(eventId: string): UseCalendarResult {
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
    setIsLoading(true);

    fetchCalendarEvent(eventId, { signal: controller.signal })
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

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [eventId, nonce]);

  return { data, isLoading, error, refresh };
}
