import type { ConditionType, ControlName, ScanCondition } from '@/lib/conditions/types';
import { accessErrorFromResponse } from '@/lib/access/errors';

/**
 * SC-2 — client for POST /api/scanner/translate.
 *
 * Same-origin: proxied to the FastAPI backend through the `/api/:path*` rewrite.
 * M.I.A is a TRANSLATOR toward the closed palette — she is never a judge. The
 * server re-validates her output against the palette BEFORE it returns, so a
 * condition here is always representable and always POST-able to the scan as-is.
 * This client never guesses: it renders exactly what the server understood, what
 * it could not translate, and what it assumed.
 */

const ENDPOINT = '/api/scanner/translate';
const DEFAULT_TIMEOUT_MS = 20_000;

export type TranslateOutcome =
  | 'translated' // every fragment recognised
  | 'partial' // some recognised, some explicitly not
  | 'none' // nothing usable understood
  | 'refused' // ranking / prediction / advice → not translated
  | 'empty' // blank input
  | 'error'; // LLM/transport failure — the textbox stays usable

export type RefusalKind = 'ranking' | 'prediction' | 'recommendation';

export interface TranslateRefusal {
  kind: RefusalKind;
}

export interface TranslateAssumption {
  condition_type: ConditionType;
  control: ControlName;
  value: string | null;
  source_phrase: string | null;
}

export interface TranslateUntranslatable {
  fragment: string | null;
  category: string;
}

export interface TranslateResult {
  outcome: TranslateOutcome;
  refusal: TranslateRefusal | null;
  /** Wire-shaped conditions, ready to POST verbatim to /api/conditions-scan. */
  conditions: ScanCondition[];
  assumptions: TranslateAssumption[];
  untranslatable: TranslateUntranslatable[];
}

export class TranslateUnavailableError extends Error {
  readonly code = 'translate_unavailable';
  constructor(message: string) {
    super(message);
    this.name = 'TranslateUnavailableError';
  }
}

export class TranslateError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'TranslateError';
  }
}

export interface TranslateOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

export async function translateStrategy(
  text: string,
  locale: string,
  options: TranslateOptions = {},
): Promise<TranslateResult> {
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

  const lang = locale.toLowerCase().startsWith('en') ? 'en' : 'fr';

  let res: Response;
  try {
    res = await fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'application/json' },
      body: JSON.stringify({ text, locale: lang }),
      signal: controller.signal,
    });
  } catch (err) {
    const message = timedOut ? 'timeout' : err instanceof Error ? err.message : 'network';
    throw new TranslateError(0, message);
  } finally {
    clearTimeout(timer);
    callerSignal?.removeEventListener('abort', onCallerAbort);
  }

  if (res.status === 503) {
    const detail = await readErrorDetail(res);
    throw new TranslateUnavailableError(detail ?? 'translator_unavailable');
  }
  // 401/402 — freemium gate (the scanner it feeds is a paid feature).
  const accessErr = await accessErrorFromResponse(res);
  if (accessErr) throw accessErr;

  if (!res.ok) {
    throw new TranslateError(res.status, 'translate_failed');
  }

  let parsed: unknown;
  try {
    parsed = await res.json();
  } catch {
    throw new TranslateError(res.status, 'unreadable');
  }
  if (!isTranslateResult(parsed)) {
    throw new TranslateError(res.status, 'malformed');
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

function isTranslateResult(value: unknown): value is TranslateResult {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.outcome === 'string' &&
    Array.isArray(v.conditions) &&
    Array.isArray(v.assumptions) &&
    Array.isArray(v.untranslatable)
  );
}
