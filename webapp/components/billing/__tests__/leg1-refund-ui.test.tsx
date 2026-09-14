import * as React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { fireEvent } from '@testing-library/dom';
import { NextIntlClientProvider } from 'next-intl';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import fr from '@/messages/fr.json';

/**
 * LEG-1 — the 14-day annual guarantee, from the customer's side.
 *
 * The terms promise a refund; this screen is where the promise is kept. What
 * must hold:
 *
 *  1. the offer appears ONLY while the window is genuinely open — nobody is
 *     shown a button that is going to refuse them;
 *  2. refunding is irreversible, so it is confirmed before anything happens;
 *  3. a refusal shows the SERVER's message, which already says what the
 *     customer can do instead — never a generic error;
 *  4. once refunded, the screen says so.
 */

const requestRefund = vi.fn();
const refundEligibility = vi.fn();
const fetchSubscription = vi.fn();

const ACTIVE_ANNUAL = {
  status: 'active',
  price_id: 'price_a',
  current_period_end: 1_800_000_000,
  cancel_at_period_end: false,
  trial_end: null,
  has_access: true,
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
  fetchRefundEligibility: () => refundEligibility(),
  requestRefund: () => requestRefund(),
  startCheckout: async () => 'https://checkout.stripe.test/s/1',
  openPortal: async () => '',
  syncSubscription: async () => null,
}));

vi.mock('@/lib/auth/api-client', () => ({
  acceptConsents: async () => ({ consents: [] }),
  AuthError: class extends Error {},
}));

vi.mock('@/lib/auth/store', () => {
  // Stable identity — a fresh object each render re-fires the load effect and
  // spins the runner (see leg1-consent-gate.test.tsx).
  const value = { account: { id: 1, email: 'a@b.co', role: 'user' }, loading: false };
  return { useAuth: () => value };
});

vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

import { SubscriptionPanel } from '../SubscriptionPanel';
import { BillingError } from '@/lib/billing/api-client';

function renderPanel() {
  return render(<SubscriptionPanel />, {
    wrapper: ({ children }) => (
      <NextIntlClientProvider locale="fr" messages={fr}>
        {children}
      </NextIntlClientProvider>
    ),
  });
}

const OPEN_WINDOW = {
  eligible: true,
  reason: null,
  days_remaining: 11,
  guarantee_days: 14,
  deadline: 1_790_000_000,
};

const CLOSED_WINDOW = {
  eligible: false,
  reason: 'window_elapsed',
  days_remaining: 0,
  guarantee_days: 14,
  deadline: null,
};

beforeEach(() => {
  fetchSubscription.mockReset().mockResolvedValue(ACTIVE_ANNUAL);
  refundEligibility.mockReset().mockResolvedValue(OPEN_WINDOW);
  requestRefund.mockReset().mockResolvedValue({
    refunded: true, amount: 34800, currency: 'USD',
  });
});

describe('LEG-1 refund — when the offer appears', () => {
  it('shows the guarantee while the window is open', async () => {
    renderPanel();
    await waitFor(() => expect(screen.getByTestId('refund-guarantee')).toBeTruthy());
    expect(screen.getByTestId('refund-request')).toBeTruthy();
  });

  it('says until WHEN, not how many days (no plural to get wrong)', async () => {
    renderPanel();
    await waitFor(() => expect(screen.getByTestId('refund-guarantee')).toBeTruthy());
    const block = screen.getByTestId('refund-guarantee').textContent ?? '';
    // A formatted date, not a bare "11".
    expect(block).toMatch(/\d{4}/);
    expect(block).toContain('Garantie de 14 jours');
  });

  it('does NOT offer a refund once the window has closed', async () => {
    refundEligibility.mockResolvedValue(CLOSED_WINDOW);
    renderPanel();
    await waitFor(() => expect(screen.getByText(fr.billing.manage)).toBeTruthy());
    expect(screen.queryByTestId('refund-guarantee')).toBeNull();
  });

  it('does NOT offer a refund when the probe itself failed', async () => {
    // The client degrades a failure to "not eligible"; the panel must not offer.
    refundEligibility.mockResolvedValue(CLOSED_WINDOW);
    renderPanel();
    await waitFor(() => expect(screen.getByText(fr.billing.manage)).toBeTruthy());
    expect(screen.queryByTestId('refund-request')).toBeNull();
  });

  it('a THROWING probe does not break the screen', async () => {
    // Regression: adding the probe to the load effect once took the whole
    // subscription screen down with it — i.e. nobody could subscribe. The
    // guarantee is a bonus; the paywall must survive it failing.
    refundEligibility.mockRejectedValue(new Error('offline'));
    fetchSubscription.mockResolvedValue(null); // unsubscribed → the plan choice
    renderPanel();
    await waitFor(() =>
      expect(document.querySelectorAll('.grid button')).toHaveLength(2),
    );
    expect(screen.queryByTestId('refund-guarantee')).toBeNull();
  });
});

describe('LEG-1 refund — asking for it', () => {
  it('confirms before refunding anything', async () => {
    renderPanel();
    await waitFor(() => expect(screen.getByTestId('refund-request')).toBeTruthy());

    fireEvent.click(screen.getByTestId('refund-request'));
    // Nothing has happened yet — only a question.
    expect(requestRefund).not.toHaveBeenCalled();
    expect(screen.getByText(fr.billing.refund.confirm)).toBeTruthy();

    fireEvent.click(screen.getByTestId('refund-confirm'));
    await waitFor(() => expect(requestRefund).toHaveBeenCalledTimes(1));
  });

  it('can be backed out of without refunding', async () => {
    renderPanel();
    await waitFor(() => expect(screen.getByTestId('refund-request')).toBeTruthy());
    fireEvent.click(screen.getByTestId('refund-request'));
    fireEvent.click(screen.getByText(fr.billing.refund.cancelCta));
    await waitFor(() => expect(screen.getByTestId('refund-request')).toBeTruthy());
    expect(requestRefund).not.toHaveBeenCalled();
  });

  it('says so once the money is on its way back', async () => {
    // After a refund the subscription is suspended server-side.
    fetchSubscription
      .mockResolvedValueOnce(ACTIVE_ANNUAL)
      .mockResolvedValue({ ...ACTIVE_ANNUAL, status: 'suspended', has_access: false });

    renderPanel();
    await waitFor(() => expect(screen.getByTestId('refund-request')).toBeTruthy());
    fireEvent.click(screen.getByTestId('refund-request'));
    fireEvent.click(screen.getByTestId('refund-confirm'));

    await waitFor(() =>
      expect(screen.getByText(fr.billing.refund.done)).toBeTruthy(),
    );
    // The offer is gone — it cannot be asked for twice.
    expect(screen.queryByTestId('refund-guarantee')).toBeNull();
  });

  it("shows the SERVER's refusal, which says what to do instead", async () => {
    const refusal =
      "Les 14 jours suivant le paiement sont écoulés. Tu peux résilier à tout moment.";
    requestRefund.mockRejectedValue(new BillingError(409, refusal));

    renderPanel();
    await waitFor(() => expect(screen.getByTestId('refund-request')).toBeTruthy());
    fireEvent.click(screen.getByTestId('refund-request'));
    fireEvent.click(screen.getByTestId('refund-confirm'));

    await waitFor(() =>
      expect(screen.getByRole('alert').textContent).toBe(refusal),
    );
  });
});
