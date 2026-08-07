import type { Account, LoginInput, RegisterInput } from './types';

/**
 * Auth API client — talks to the FastAPI account routes through the same-origin
 * `/api/*` rewrite (next.config.js). The session is a first-party HttpOnly
 * cookie set by the backend, so requests just need `credentials: 'same-origin'`
 * (the default) — there is no token to attach by hand and none is exposed to JS.
 */

const BASE = '/api/auth';

/**
 * Per-attempt timeout for every auth call (AUTH-HARDENING). The login/register/
 * reset fetches previously had NO timeout: a hung backend left the form stuck on
 * "Connexion…" forever with no error — the same defect REC-1 fixed for the access
 * gate, but never for the auth forms. 8 s mirrors the access gate and the
 * market-reading client.
 */
const DEFAULT_TIMEOUT_MS = 8_000;

/**
 * Why an auth request failed, so the form can say WHICH honestly instead of one
 * vague line:
 *   · 'timeout'  → our request budget elapsed (server slow / not answering)
 *   · 'network'  → the connection itself failed (server unreachable / offline)
 *   · 'http'     → the backend answered with a 4xx/5xx (message is its `detail`)
 */
export type AuthErrorReason = 'timeout' | 'network' | 'http';

/** A failed auth request with a user-safe message + HTTP status (0 = transport). */
export class AuthError extends Error {
  readonly status: number;
  readonly reason: AuthErrorReason;
  constructor(status: number, message: string, reason: AuthErrorReason = 'http') {
    super(message);
    this.status = status;
    this.reason = reason;
    this.name = 'AuthError';
  }
}

async function readDetail(res: Response): Promise<string | null> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body?.detail === 'string') return body.detail;
  } catch {
    /* fall through */
  }
  return null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  try {
    return await attempt<T>(path, init);
  } catch (err) {
    // Retry once ONLY on a genuine network transient — never on a TIMEOUT (the
    // budget is already spent; retrying doubles the wait) nor a deterministic HTTP
    // error (a 401/409/429 won't change on a blind retry). Mirrors the reading
    // client's policy.
    if (err instanceof AuthError && err.reason === 'network') {
      return attempt<T>(path, init);
    }
    throw err;
  }
}

async function attempt<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, DEFAULT_TIMEOUT_MS);

  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { 'content-type': 'application/json' },
      // Send the HttpOnly session cookie. `same-origin` is the fetch default,
      // but we set it explicitly so the auth flow keeps working if these calls
      // are ever proxied, and to document the dependency (AUTH-11: the whole
      // session relies on this cookie riding along with every auth request).
      credentials: 'same-origin',
      signal: controller.signal,
      ...init,
    });
  } catch (err) {
    // A timed-out request and a genuine network failure both land here (fetch
    // rejects with an AbortError on timeout). Tag which so the form can say
    // "trop lent" vs "injoignable" — never a frozen "Connexion…" with no message.
    if (timedOut) {
      throw new AuthError(
        0,
        'Le serveur met trop de temps à répondre. Réessayez dans un instant.',
        'timeout',
      );
    }
    const message = err instanceof Error ? err.message : 'Erreur réseau';
    throw new AuthError(0, `Serveur injoignable : ${message}`, 'network');
  } finally {
    clearTimeout(timer);
  }

  if (res.status === 204) return undefined as T;

  let parsed: unknown = null;
  const text = await res.text();
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = null;
    }
  }

  if (!res.ok) {
    const detail =
      (parsed as { detail?: unknown } | null)?.detail;
    const message =
      typeof detail === 'string'
        ? detail
        : 'Une erreur est survenue. Réessaie dans un instant.';
    throw new AuthError(res.status, message);
  }

  return parsed as T;
}

export function register(input: RegisterInput): Promise<Account> {
  return request<Account>('/register', {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

export function login(input: LoginInput): Promise<Account> {
  return request<Account>('/login', {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

export function logout(): Promise<{ ok: boolean; message: string }> {
  return request('/logout', { method: 'POST' });
}

/** Current account, or null when not authenticated. 401 (no/expired session)
 *  and 403 (account deactivated — AUTH-16) both mean "not logged in" here, not a
 *  transport error, so the provider treats them as anonymous rather than a
 *  failed probe. */
export async function fetchMe(): Promise<Account | null> {
  try {
    return await request<Account>('/me', { method: 'GET' });
  } catch (err) {
    if (err instanceof AuthError && (err.status === 401 || err.status === 403)) {
      return null;
    }
    throw err;
  }
}

export function updateProfile(email: string): Promise<Account> {
  return request<Account>('/profile', {
    method: 'PATCH',
    body: JSON.stringify({ email }),
  });
}

export function requestPasswordReset(
  identifier: string,
): Promise<{ ok: boolean; message: string }> {
  return request('/password-reset/request', {
    method: 'POST',
    body: JSON.stringify({ identifier }),
  });
}

export function confirmPasswordReset(
  token: string,
  newPassword: string,
): Promise<{ ok: boolean; message: string }> {
  return request('/password-reset/confirm', {
    method: 'POST',
    body: JSON.stringify({ token, new_password: newPassword }),
  });
}

export { readDetail };
