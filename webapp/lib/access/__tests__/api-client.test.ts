import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchAccess } from '../api-client';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('fetchAccess', () => {
  it('sends the same-origin session cookie with the access probe', async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ authenticated: true }), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await fetchAccess();

    // The gate decision depends on the HttpOnly session cookie riding along;
    // credentials must be set explicitly (parity with the auth client).
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/access/me',
      expect.objectContaining({ credentials: 'same-origin' }),
    );
  });

  it('throws on a transport/parse failure (non-OK response)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('boom', { status: 500 })));
    await expect(fetchAccess()).rejects.toThrow(/access summary unavailable/);
  });
});
