import type { CalendarResponse } from '@/types/calendar';

/**
 * Client for GET /api/calendar (NW-1). Same-origin: proxied to FastAPI through
 * the `/api/:path*` rewrite, so the browser never talks to the backend host.
 *
 * The endpoint is descriptive-only. The active source is env-driven server-side
 * and defaults to the official stub (0 events) until official feeds are wired.
 */

const ENDPOINT = '/api/calendar';
const DEFAULT_TIMEOUT_MS = 8_000;

export class CalendarError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'CalendarError';
  }
}

export interface FetchCalendarOptions {
  signal?: AbortSignal;
  lookaheadDays?: number;
  lookbackDays?: number;
  timeoutMs?: number;
}

export async function fetchCalendar(
  options: FetchCalendarOptions = {},
): Promise<CalendarResponse> {
  const {
    signal,
    lookaheadDays = 7,
    lookbackDays = 3,
    timeoutMs = DEFAULT_TIMEOUT_MS,
  } = options;

  const url =
    `${ENDPOINT}?lookahead_days=${encodeURIComponent(String(lookaheadDays))}` +
    `&lookback_days=${encodeURIComponent(String(lookbackDays))}`;

  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onCallerAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener('abort', onCallerAbort, { once: true });
  }

  let res: Response;
  try {
    res = await fetch(url, {
      method: 'GET',
      headers: { accept: 'application/json' },
      signal: controller.signal,
    });
  } catch (err) {
    const message = timedOut
      ? 'Délai dépassé en interrogeant le calendrier.'
      : err instanceof Error
        ? err.message
        : 'Erreur réseau';
    throw new CalendarError(0, `Service de calendrier injoignable : ${message}`);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onCallerAbort);
  }

  if (!res.ok) {
    throw new CalendarError(
      res.status,
      'Le calendrier a rencontré une erreur. Réessaie dans un instant.',
    );
  }

  let parsed: unknown;
  try {
    parsed = await res.json();
  } catch {
    throw new CalendarError(res.status, 'Réponse du calendrier illisible.');
  }

  if (!isCalendarShape(parsed)) {
    throw new CalendarError(res.status, 'Réponse du calendrier malformée.');
  }
  return parsed;
}

/**
 * Fetch ONE event by its stable id, independent of any window (REC point 1).
 * Returns a CalendarResponse whose `events` holds the matching event, or is
 * empty when the id genuinely does not exist. Same timeout/shape guarantees.
 */
export async function fetchCalendarEvent(
  eventId: string,
  options: { signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<CalendarResponse> {
  const { signal, timeoutMs = DEFAULT_TIMEOUT_MS } = options;
  const url = `${ENDPOINT}/event/${encodeURIComponent(eventId)}`;

  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onCallerAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener('abort', onCallerAbort, { once: true });
  }

  let res: Response;
  try {
    res = await fetch(url, {
      method: 'GET',
      headers: { accept: 'application/json' },
      signal: controller.signal,
    });
  } catch (err) {
    const message = timedOut
      ? 'Délai dépassé en interrogeant le calendrier.'
      : err instanceof Error
        ? err.message
        : 'Erreur réseau';
    throw new CalendarError(0, `Service de calendrier injoignable : ${message}`);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onCallerAbort);
  }

  if (!res.ok) {
    throw new CalendarError(
      res.status,
      'Le calendrier a rencontré une erreur. Réessaie dans un instant.',
    );
  }

  let parsed: unknown;
  try {
    parsed = await res.json();
  } catch {
    throw new CalendarError(res.status, 'Réponse du calendrier illisible.');
  }
  if (!isCalendarShape(parsed)) {
    throw new CalendarError(res.status, 'Réponse du calendrier malformée.');
  }
  return parsed;
}

function isCalendarShape(value: unknown): value is CalendarResponse {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    Array.isArray(v.events) &&
    typeof v.coverage === 'object' &&
    v.coverage !== null &&
    typeof v.window_start === 'string' &&
    typeof v.window_end === 'string'
  );
}
