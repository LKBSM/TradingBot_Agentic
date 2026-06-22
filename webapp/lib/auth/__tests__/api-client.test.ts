import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AuthError,
  confirmPasswordReset,
  fetchMe,
  login,
  logout,
  register,
  requestPasswordReset,
} from '@/lib/auth/api-client';

/** Unit tests for the auth api-client. `fetch` is fully mocked (no network). */

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

const ACCOUNT = {
  id: 1,
  username: 'alice',
  email: 'alice@example.com',
  role: 'user' as const,
  age_confirmed: true,
  created_at: '2026-06-22T00:00:00',
  consents: [{ doc: 'terms', version: '2026-04-28', accepted_at: '2026-06-22T00:00:00' }],
};

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('register', () => {
  it('POSTs to /api/auth/register and returns the account', async () => {
    fetchMock.mockResolvedValue(jsonResponse(201, ACCOUNT));
    const acc = await register({
      username: 'alice',
      email: 'alice@example.com',
      password: 'longpassword1',
      age_confirmed: true,
      accept_terms: true,
      accept_privacy: true,
    });
    expect(acc.username).toBe('alice');
    const [url, init] = fetchMock.mock.calls.at(-1)!;
    expect(url).toBe('/api/auth/register');
    expect((init as RequestInit).method).toBe('POST');
  });

  it('throws AuthError with backend detail on 409', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(409, { detail: 'Ce nom d’utilisateur est déjà pris.' }),
    );
    await expect(
      register({
        username: 'alice',
        email: 'a@b.co',
        password: 'longpassword1',
        age_confirmed: true,
        accept_terms: true,
        accept_privacy: true,
      }),
    ).rejects.toMatchObject({ status: 409, name: 'AuthError' });
  });
});

describe('login', () => {
  it('returns the account on 200', async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, ACCOUNT));
    const acc = await login({ identifier: 'alice', password: 'longpassword1' });
    expect(acc.id).toBe(1);
  });

  it('throws AuthError(401) on bad credentials', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(401, { detail: 'Identifiant ou mot de passe incorrect.' }),
    );
    await expect(
      login({ identifier: 'alice', password: 'nope' }),
    ).rejects.toBeInstanceOf(AuthError);
  });
});

describe('fetchMe', () => {
  it('returns the account when authenticated', async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, ACCOUNT));
    expect(await fetchMe()).toMatchObject({ username: 'alice' });
  });

  it('returns null on 401 (not an error)', async () => {
    fetchMock.mockResolvedValue(jsonResponse(401, { detail: 'Authentication required' }));
    expect(await fetchMe()).toBeNull();
  });

  it('rethrows non-401 failures', async () => {
    fetchMock.mockResolvedValue(jsonResponse(500, { detail: 'boom' }));
    await expect(fetchMe()).rejects.toBeInstanceOf(AuthError);
  });
});

describe('logout', () => {
  it('POSTs to /api/auth/logout', async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, { ok: true, message: 'Déconnecté.' }));
    await logout();
    expect(fetchMock.mock.calls.at(-1)![0]).toBe('/api/auth/logout');
  });
});

describe('password reset', () => {
  it('request posts the identifier', async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, { ok: true, message: 'ok' }));
    await requestPasswordReset('alice');
    const init = fetchMock.mock.calls.at(-1)![1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({ identifier: 'alice' });
  });

  it('confirm posts token + new_password', async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, { ok: true, message: 'ok' }));
    await confirmPasswordReset('tok123', 'brandnewpass1');
    const init = fetchMock.mock.calls.at(-1)![1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({
      token: 'tok123',
      new_password: 'brandnewpass1',
    });
  });
});

describe('network failure', () => {
  it('wraps a thrown fetch into AuthError(status=0)', async () => {
    fetchMock.mockRejectedValue(new Error('ECONNREFUSED'));
    await expect(login({ identifier: 'a', password: 'b' })).rejects.toMatchObject({
      status: 0,
    });
  });
});
