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

  it('REC-1: aborts on timeout so the gate can never spin forever', async () => {
    // A backend that never responds must not leave the SubscriptionGate (on every
    // product page) stuck on the loading skeleton. fetchAccess bounds the wait:
    // the fetch is passed an AbortSignal that trips on timeout → it rejects.
    vi.useFakeTimers();
    const fetchMock = vi.fn(
      (_url: string, init: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          // Never resolves; only rejects when the internal timeout aborts it.
          init.signal?.addEventListener('abort', () =>
            reject(new DOMException('aborted', 'AbortError')),
          );
        }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const promise = fetchAccess();
    const assertion = expect(promise).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(8_000); // ACCESS_TIMEOUT_MS
    await assertion;
    expect(fetchMock).toHaveBeenCalled();
    vi.useRealTimers();
  });
});
