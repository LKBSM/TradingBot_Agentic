import type { ConditionsConfig, ConditionsScanResponse } from './types';

/**
 * Client for POST /api/conditions-scan — the read-only structural scan.
 *
 * Same-origin: proxied to the FastAPI backend through the `/api/:path*` rewrite
 * (next.config.js). The scan is descriptive and read-only on the server — this
 * client only sends the user's conditions and renders the per-combo breakdown.
 *
 * Backend contract (src/api/routes/conditions_scan.py):
 *   · 200 → ConditionsScanResponse
 *   · 422 → invalid conditions (e.g. unknown / predictive type) → ScanValidationError
 *   · 503 → service not configured                              → ScanNotAvailableError
 *   · 500 / network                                            → ScanError
 */

const ENDPOINT = '/api/conditions-scan';
const DEFAULT_TIMEOUT_MS = 8_000;

export class ScanNotAvailableError extends Error {
  readonly code = 'scan_unavailable';
  constructor(message: string) {
    super(message);
    this.name = 'ScanNotAvailableError';
  }
}

export class ScanValidationError extends Error {
  readonly status = 422;
  constructor(message: string) {
    super(message);
    this.name = 'ScanValidationError';
  }
}

export class ScanError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'ScanError';
  }
}

export interface FetchScanOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

export async function fetchConditionsScan(
  config: ConditionsConfig,
  options: FetchScanOptions = {},
): Promise<ConditionsScanResponse> {
  const { signal: callerSignal, timeoutMs = DEFAULT_TIMEOUT_MS } = options;

  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  const onCallerAbort = () => controller.abort();
  if (callerSignal) {
    if (callerSignal.aborted) controller.abort();
    else callerSignal.addEventListener('abort', onCallerAbort, { once: true });
  }

  let res: Response;
  try {
    res = await fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'application/json' },
      body: JSON.stringify(config),
      signal: controller.signal,
    });
  } catch (err) {
    const message = timedOut
      ? 'Délai dépassé en lançant le scan.'
      : err instanceof Error
        ? err.message
        : 'Erreur réseau';
    throw new ScanError(0, `Service de scan injoignable : ${message}`);
  } finally {
    clearTimeout(timer);
    callerSignal?.removeEventListener('abort', onCallerAbort);
  }

  if (res.status === 422) {
    throw new ScanValidationError('Conditions invalides.');
  }
  if (res.status === 503) {
    const detail = await readErrorDetail(res);
    throw new ScanNotAvailableError(
      detail ?? "Le service de scan n’est pas disponible sur cet environnement.",
    );
  }
  if (!res.ok) {
    throw new ScanError(
      res.status,
      'Le service de scan a rencontré une erreur interne. Réessaie dans un instant.',
    );
  }

  let parsed: unknown;
  try {
    parsed = await res.json();
  } catch {
    throw new ScanError(res.status, 'Réponse du service illisible.');
  }
  if (!isScanResponseShape(parsed)) {
    throw new ScanError(res.status, 'Réponse du service malformée.');
  }
  return parsed;
}

async function readErrorDetail(res: Response): Promise<string | null> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    return typeof body?.detail === 'string' ? body.detail : null;
  } catch {
    return null;
  }
}

function isScanResponseShape(value: unknown): value is ConditionsScanResponse {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return Array.isArray(v.matches) && Array.isArray(v.unavailable);
}
