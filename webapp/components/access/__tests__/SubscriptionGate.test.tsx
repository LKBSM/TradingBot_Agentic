import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { SubscriptionGate } from '../SubscriptionGate';
import { comboAllowed, type AccessSummary } from '@/lib/access/api-client';
import { accessErrorFromResponse } from '@/lib/access/errors';

const hoisted = vi.hoisted(() => ({
  pathname: '/app',
  replace: vi.fn(),
}));
vi.mock('next/navigation', () => ({
  usePathname: () => hoisted.pathname,
  useRouter: () => ({ push: vi.fn(), replace: hoisted.replace }),
}));

function stubAccess(summary: AccessSummary) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      new Response(JSON.stringify(summary), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    ),
  );
}

const FULL: AccessSummary = {
  authenticated: true,
  gate_enforced: true,
  beta_lockdown: false,
  must_login: false,
  tier: 'subscriber',
  is_owner: false,
  has_full_access: true,
  entitlements: {
    instruments: null,
    timeframes: null,
    scanner: true,
    chat: { limit: null, used: null, remaining: null },
  },
};

const FREE: AccessSummary = {
  authenticated: true,
  gate_enforced: true,
  beta_lockdown: false,
  must_login: false,
  tier: 'free',
  is_owner: false,
  has_full_access: false,
  entitlements: {
    instruments: ['XAUUSD'],
    timeframes: ['M15'],
    scanner: false,
    chat: { limit: 5, used: 0, remaining: 5 },
  },
};

const VISITOR: AccessSummary = {
  authenticated: false,
  gate_enforced: true,
  beta_lockdown: false,
  must_login: false,
  tier: 'visitor',
  is_owner: false,
  has_full_access: false,
  entitlements: {
    instruments: ['XAUUSD'],
    timeframes: ['M15'],
    scanner: false,
    chat: { limit: 5, used: null, remaining: null },
  },
};

// Closed beta, anonymous caller: gate not enforced (freemium off) but the beta
// lockdown demands login. must_login drives the redirect independently.
const LOCKDOWN_ANON: AccessSummary = {
  authenticated: false,
  gate_enforced: false,
  beta_lockdown: true,
  must_login: true,
  tier: 'visitor',
  is_owner: false,
  has_full_access: false,
  entitlements: {
    instruments: null,
    timeframes: null,
    scanner: true,
    chat: { limit: null, used: null, remaining: null },
  },
};

afterEach(() => {
  hoisted.pathname = '/app';
  hoisted.replace.mockReset();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('SubscriptionGate', () => {
  it('renders children for a full-access account', async () => {
    stubAccess(FULL);
    render(
      <SubscriptionGate>
        <div>secret content</div>
      </SubscriptionGate>,
    );
    expect(await screen.findByText('secret content')).toBeInTheDocument();
  });

  it('redirects an unauthenticated visitor to /connexion when enforced', async () => {
    stubAccess(VISITOR);
    render(
      <SubscriptionGate>
        <div>secret content</div>
      </SubscriptionGate>,
    );
    await waitFor(() =>
      expect(hoisted.replace).toHaveBeenCalledWith(
        '/connexion?next=%2Fapp',
      ),
    );
    expect(screen.queryByText('secret content')).not.toBeInTheDocument();
  });

  it('redirects to /connexion under beta lockdown (must_login)', async () => {
    stubAccess(LOCKDOWN_ANON);
    render(
      <SubscriptionGate>
        <div>secret content</div>
      </SubscriptionGate>,
    );
    await waitFor(() =>
      expect(hoisted.replace).toHaveBeenCalledWith('/connexion?next=%2Fapp'),
    );
    expect(screen.queryByText('secret content')).not.toBeInTheDocument();
  });

  it('shows a paywall for a free account on a paid-only surface', async () => {
    stubAccess(FREE);
    render(
      <SubscriptionGate requireFullAccess paywallTitle="Réservé">
        <div>scanner content</div>
      </SubscriptionGate>,
    );
    expect(await screen.findByText('Réservé')).toBeInTheDocument();
    expect(screen.queryByText('scanner content')).not.toBeInTheDocument();
    // The paywall invites subscription.
    expect(screen.getByText('Voir les abonnements')).toBeInTheDocument();
  });

  it('lets a free account into a partial surface (no requireFullAccess)', async () => {
    stubAccess(FREE);
    render(
      <SubscriptionGate>
        <div>reading content</div>
      </SubscriptionGate>,
    );
    expect(await screen.findByText('reading content')).toBeInTheDocument();
  });

  it('fails open (renders children) when the summary fetch errors', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('boom', { status: 500 })));
    render(
      <SubscriptionGate>
        <div>fallback content</div>
      </SubscriptionGate>,
    );
    expect(await screen.findByText('fallback content')).toBeInTheDocument();
  });
});

describe('comboAllowed', () => {
  it('allows everything for full access', () => {
    expect(comboAllowed(FULL, 'EURUSD', 'H4')).toBe(true);
  });
  it('restricts a free account to its perimeter', () => {
    expect(comboAllowed(FREE, 'XAUUSD', 'M15')).toBe(true);
    expect(comboAllowed(FREE, 'XAUUSD', 'H1')).toBe(false);
    expect(comboAllowed(FREE, 'EURUSD', 'M15')).toBe(false);
  });
});

describe('accessErrorFromResponse', () => {
  it('maps 402 to a subscription upsell error', async () => {
    const res = new Response(JSON.stringify({ detail: 'Abonnement requis.' }), {
      status: 402,
      headers: { 'content-type': 'application/json' },
    });
    const err = await accessErrorFromResponse(res);
    expect(err?.status).toBe(402);
    expect(err?.needsLogin).toBe(false);
    expect(err?.message).toBe('Abonnement requis.');
  });

  it('maps 401 to a needs-login error', async () => {
    const res = new Response(null, { status: 401 });
    const err = await accessErrorFromResponse(res);
    expect(err?.status).toBe(401);
    expect(err?.needsLogin).toBe(true);
  });

  it('returns null for non-access statuses', async () => {
    const res = new Response(null, { status: 503 });
    expect(await accessErrorFromResponse(res)).toBeNull();
  });
});
