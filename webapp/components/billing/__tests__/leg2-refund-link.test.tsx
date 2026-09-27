import * as React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { fireEvent } from '@testing-library/dom';
import { NextIntlClientProvider } from 'next-intl';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import fr from '@/messages/fr.json';

/**
 * LEG-2 — on /abonnement, two things must hold next to the buttons that take the
 * money:
 *
 *   1. the way OUT is stated where the decision is made — a link to the
 *      cancellation/refund clause, not three clicks away in the footer;
 *   2. « Gérer mon abonnement » really opens the Stripe portal, i.e. it calls the
 *      portal endpoint and follows the URL Stripe returns. This is the button the
 *      terms now tell the customer to click to cancel, so it is a promise in the
 *      contract, not just a convenience.
 */

const openPortal = vi.fn();
const fetchSubscription = vi.fn();

const ACTIVE_ANNUAL = {
  status: 'active',
  price_id: 'price_a',
  current_period_end: 1_800_000_000,
  cancel_at_period_end: false,
  trial_end: null,
  has_access: true,
};

const NO_SUBSCRIPTION = {
  status: null,
  price_id: null,
  current_period_end: null,
  cancel_at_period_end: false,
  trial_end: null,
  has_access: false,
};

vi.mock('@/lib/billing/api-client', () => ({
  BillingError: class BillingError extends Error {
    status: number;
    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
  fetchPricing: async () => ({
    plans: [
      { key: 'MONTHLY', price_id: 'price_m' },
      { key: 'ANNUAL', price_id: 'price_a' },
    ],
  }),
  fetchSubscription: () => fetchSubscription(),
  fetchRefundEligibility: async () => null,
  requestRefund: async () => ({ refunded: true }),
  startCheckout: async () => 'https://checkout.stripe.test/s/1',
  openPortal: () => openPortal(),
  syncSubscription: async () => null,
}));

vi.mock('@/lib/auth/api-client', () => ({
  acceptConsents: async () => ({ consents: [] }),
  AuthError: class extends Error {},
}));

vi.mock('@/lib/auth/store', () => {
  // Stable identity — a fresh object each render re-fires the load effect.
  const value = { account: { id: 1, email: 'a@b.co', role: 'user' }, loading: false };
  return { useAuth: () => value };
});

vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

import { SubscriptionPanel } from '../SubscriptionPanel';

function renderPanel() {
  return render(<SubscriptionPanel />, {
    wrapper: ({ children }) => (
      <NextIntlClientProvider locale="fr" messages={fr}>
        {children}
      </NextIntlClientProvider>
    ),
  });
}

beforeEach(() => {
  openPortal.mockReset();
  fetchSubscription.mockReset();
});

describe('LEG-2 — the refund clause is linked where the payment happens', () => {
  it('links to /conditions#remboursement from the plan-choice screen', async () => {
    fetchSubscription.mockResolvedValue(NO_SUBSCRIPTION);
    renderPanel();

    const link = await screen.findByRole('link', {
      name: fr.billing.refundPolicyLink,
    });
    expect(link).toHaveAttribute('href', '/conditions#remboursement');
  });
});

describe('LEG-2 — « Gérer mon abonnement » opens the Stripe portal', () => {
  it('asks the backend for a portal session and follows its URL', async () => {
    fetchSubscription.mockResolvedValue(ACTIVE_ANNUAL);
    openPortal.mockResolvedValue('https://billing.stripe.test/session/abc');
    renderPanel();

    fireEvent.click(await screen.findByRole('button', { name: fr.billing.manage }));

    // The portal session is minted server-side (POST /api/billing/portal — see
    // `api-client-portal.test.ts` for the endpoint, and tests/test_account_billing.py
    // for Stripe). The redirect itself (`window.location.href = url`) is not
    // observable in jsdom, so what is asserted here is that the click reaches it.
    await waitFor(() => expect(openPortal).toHaveBeenCalledTimes(1));
  });

  it('says so, rather than failing silently, when the portal cannot open', async () => {
    fetchSubscription.mockResolvedValue(ACTIVE_ANNUAL);
    openPortal.mockRejectedValue(new Error('boom'));
    renderPanel();

    fireEvent.click(await screen.findByRole('button', { name: fr.billing.manage }));

    expect(await screen.findByText(fr.billing.errorPortal)).toBeInTheDocument();
  });
});
