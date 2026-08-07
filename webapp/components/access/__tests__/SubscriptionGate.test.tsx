import { render, screen, waitFor } from '@/components/test-utils';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SubscriptionGate } from '../SubscriptionGate';
import { type AccessSummary } from '@/lib/access/api-client';
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
  is_owner: false,
  has_access: true,
  subscription_required: false,
};

// Authenticated but no active subscription — the paywalled state (PAY-1).
const UNSUBSCRIBED: AccessSummary = {
  authenticated: true,
  gate_enforced: true,
  beta_lockdown: false,
  must_login: false,
  is_owner: false,
  has_access: false,
  subscription_required: true,
};

const VISITOR: AccessSummary = {
  authenticated: false,
  gate_enforced: true,
  beta_lockdown: false,
  must_login: false,
  is_owner: false,
  has_access: false,
  subscription_required: false,
};

// Closed beta, anonymous caller: gate not enforced (payment wall off) but the
// beta lockdown demands login. must_login drives the redirect independently.
const LOCKDOWN_ANON: AccessSummary = {
  authenticated: false,
  gate_enforced: false,
  beta_lockdown: true,
  must_login: true,
  is_owner: false,
  has_access: false,
  subscription_required: false,
};

afterEach(() => {
  hoisted.pathname = '/app';
  hoisted.replace.mockReset();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('SubscriptionGate', () => {
  it('renders children for a subscribed account', async () => {
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

  it('shows a paywall for an unsubscribed account on any gated surface', async () => {
    stubAccess(UNSUBSCRIBED);
    render(
      <SubscriptionGate paywallTitle="Réservé">
        <div>scanner content</div>
      </SubscriptionGate>,
    );
    expect(await screen.findByText('Réservé')).toBeInTheDocument();
    expect(screen.queryByText('scanner content')).not.toBeInTheDocument();
    // The paywall invites subscription.
    expect(screen.getByText('Voir les abonnements')).toBeInTheDocument();
  });

  it('lets an unsubscribed account through when requireSubscription is false', async () => {
    // The account page and the subscription page must never paywall — that is
    // exactly where an unsubscribed account lands to subscribe (PAY-1).
    stubAccess(UNSUBSCRIBED);
    render(
      <SubscriptionGate requireSubscription={false}>
        <div>account content</div>
      </SubscriptionGate>,
    );
    expect(await screen.findByText('account content')).toBeInTheDocument();
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
