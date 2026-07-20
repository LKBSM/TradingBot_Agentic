import type { Account, LoginInput, RegisterInput } from './types';

/**
 * Auth API client — talks to the FastAPI account routes through the same-origin
 * `/api/*` rewrite (next.config.js). The session is a first-party HttpOnly
 * cookie set by the backend, so requests just need `credentials: 'same-origin'`
 * (the default) — there is no token to attach by hand and none is exposed to JS.
 */

const BASE = '/api/auth';

/** A failed auth request with a user-safe message + HTTP status (0 = network). */
export class AuthError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
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
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { 'content-type': 'application/json' },
      // Send the HttpOnly session cookie. `same-origin` is the fetch default,
      // but we set it explicitly so the auth flow keeps working if these calls
      // are ever proxied, and to document the dependency (AUTH-11: the whole
      // session relies on this cookie riding along with every auth request).
      credentials: 'same-origin',
      ...init,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Erreur réseau';
    throw new AuthError(0, `Connexion impossible : ${message}`);
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
