import { afterEach, describe, expect, it, vi } from 'vitest';

import { BillingError, openPortal } from '../api-client';

/**
 * LEG-2 — the last link in the cancellation chain that can be tested in the
 * front-end: `openPortal()` must hit the account-bound billing route and hand
 * back the URL Stripe minted, so the click on « Gérer mon abonnement » lands in
 * the portal where cancelling actually happens.
 *
 * The chain: /compte → /abonnement → « Gérer mon abonnement » → this call →
 * POST /api/billing/portal (tests/test_account_billing.py) → Stripe.
 */

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('openPortal', () => {
  it('POSTs to the billing portal route and returns the Stripe URL', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      // The client reads the body as text, then parses it — mirror that.
      text: async () => JSON.stringify({ url: 'https://billing.stripe.test/session/abc' }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await expect(openPortal()).resolves.toBe('https://billing.stripe.test/session/abc');

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('/api/billing/portal');
    expect(init.method).toBe('POST');
  });

  it('surfaces the backend refusal instead of a blank failure', async () => {
    // 409 is what the route answers for an account with no Stripe customer yet.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 409,
        text: async () => JSON.stringify({ detail: 'No billing account yet' }),
      }),
    );

    await expect(openPortal()).rejects.toBeInstanceOf(BillingError);
  });
});
