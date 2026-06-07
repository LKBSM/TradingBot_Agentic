import type { MarketReading } from '@/types/market-reading';

/**
 * Client for GET /api/market-reading — Chantier 2 lazy on-demand endpoint.
 *
 * Same-origin: the request is proxied to the FastAPI backend through the
 * `/api/:path*` rewrite (next.config.js), so the browser never talks to the
 * backend host directly (CSP `connect-src 'self'`).
 *
 * Backend contract (src/api/routes/market_reading.py):
 *   · 200 → MarketReading JSON
 *   · 400 → unsupported instrument / timeframe  → MarketReadingValidationError
 *   · 503 → assembler not configured            → MarketReadingNotAvailableError
 *   · 500 → internal error                       → MarketReadingError
 */

const ENDPOINT = '/api/market-reading';
const DEFAULT_TIMEOUT_MS = 8_000;

/** 503 — the MarketReading service is not bootstrapped on this environment. */
export class MarketReadingNotAvailableError extends Error {
  readonly code = 'market_reading_unavailable';
  constructor(message: string) {
    super(message);
    this.name = 'MarketReadingNotAvailableError';
  }
}

/** 400 — instrument/timeframe outside the supported V1 perimeter. */
export class MarketReadingValidationError extends Error {
  readonly status = 400;
  constructor(message: string) {
    super(message);
    this.name = 'MarketReadingValidationError';
  }
}

/** 500 / other HTTP / network / parse failures. `status` is 0 for transport errors. */
export class MarketReadingError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'MarketReadingError';
  }
}

export interface FetchMarketReadingOptions {
  /** Caller-controlled abort handle (e.g. wired to a React effect cleanup). */
  signal?: AbortSignal;
  /** Per-attempt timeout in ms (default 8000). */
  timeoutMs?: number;
  /** Retry once on a transient transport error / timeout (default true). */
  retry?: boolean;
}

/**
 * Fetch the current market reading for a given instrument/timeframe combo.
 *
 * @throws {MarketReadingValidationError} on 400
 * @throws {MarketReadingNotAvailableError} on 503
 * @throws {MarketReadingError} on 500 / other HTTP / network / parse failures
 */
export async function fetchMarketReading(
  instrument: string,
  timeframe: string,
  options: FetchMarketReadingOptions = {},
): Promise<MarketReading> {
  const { signal, timeoutMs = DEFAULT_TIMEOUT_MS, retry = true } = options;

  try {
    return await attempt(instrument, timeframe, signal, timeoutMs);
  } catch (err) {
    // Retry once on a transient transport error or timeout — never on a
    // deterministic HTTP error (400/500/503) or a caller-initiated abort.
    const isTransient =
      err instanceof MarketReadingError && err.status === 0;
    const callerAborted = signal?.aborted ?? false;
    if (retry && isTransient && !callerAborted) {
      return attempt(instrument, timeframe, signal, timeoutMs);
    }
    throw err;
  }
}

async function attempt(
  instrument: string,
  timeframe: string,
  callerSignal: AbortSignal | undefined,
  timeoutMs: number,
): Promise<MarketReading> {
  const url = `${ENDPOINT}?instrument=${encodeURIComponent(instrument)}&timeframe=${encodeURIComponent(timeframe)}`;

  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  // Forward a caller abort to our internal controller.
  const onCallerAbort = () => controller.abort();
  if (callerSignal) {
    if (callerSignal.aborted) controller.abort();
    else callerSignal.addEventListener('abort', onCallerAbort, { once: true });
  }

  let res: Response;
  try {
    res = await fetch(url, {
      method: 'GET',
      headers: { accept: 'application/json' },
      signal: controller.signal,
    });
  } catch (err) {
    // AbortError (timeout or caller abort) and genuine network failures land
    // here. We surface them as a transient transport error (status 0).
    const message = timedOut
      ? 'Délai dépassé en interrogeant le service de lecture.'
      : err instanceof Error
        ? err.message
        : 'Erreur réseau';
    throw new MarketReadingError(0, `Service de lecture injoignable : ${message}`);
  } finally {
    clearTimeout(timer);
    callerSignal?.removeEventListener('abort', onCallerAbort);
  }

  if (res.status === 400) {
    const detail = await readErrorDetail(res);
    throw new MarketReadingValidationError(
      detail ?? 'Instrument ou timeframe non supporté.',
    );
  }

  if (res.status === 503) {
    const detail = await readErrorDetail(res);
    throw new MarketReadingNotAvailableError(
      detail ?? "Le service de lecture n'est pas disponible sur cet environnement.",
    );
  }

  if (!res.ok) {
    // 500 and anything else — never surface server internals.
    throw new MarketReadingError(
      res.status,
      'Le service de lecture a rencontré une erreur interne. Réessaie dans un instant.',
    );
  }

  let parsed: unknown;
  try {
    parsed = await res.json();
  } catch {
    throw new MarketReadingError(res.status, 'Réponse du service illisible.');
  }

  if (!isMarketReadingShape(parsed)) {
    throw new MarketReadingError(res.status, 'Réponse du service malformée.');
  }

  return parsed;
}

/** Best-effort extraction of a FastAPI `{detail}` body; never throws. */
async function readErrorDetail(res: Response): Promise<string | null> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    return typeof body?.detail === 'string' ? body.detail : null;
  } catch {
    return null;
  }
}

/** Minimal structural guard — the five top-level blocks must be present. */
function isMarketReadingShape(value: unknown): value is MarketReading {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.header === 'object' &&
    v.header !== null &&
    typeof v.structure === 'object' &&
    typeof v.regime === 'object' &&
    typeof v.events === 'object' &&
    typeof v.conditions === 'object'
  );
}
