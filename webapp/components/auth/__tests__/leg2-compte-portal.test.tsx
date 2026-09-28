import { render, waitFor, within } from '@/components/test-utils';
import { fireEvent } from '@testing-library/dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import fr from '@/messages/fr.json';
import { STRIPE_PORTAL_LOGIN_URL } from '@/lib/billing/portal';

/**
 * LEG-2 — /compte, and the promise clause 8.1 makes.
 *
 * The terms now tell the customer that cancelling happens « depuis ton espace
 * client, par le bouton …, qui ouvre le portail de facturation hébergé par
 * Stripe ». So from /compte that has to be ONE click into the portal, not a
 * detour through /abonnement.
 *
 * Three things that must hold together:
 *
 *   1. the click mints a portal session and follows it;
 *   2. an account with no Stripe customer yet (409) lands on the plan choice —
 *      there is nothing to manage, so an error about a portal would be noise;
 *   3. while the 14-day guarantee is OPEN, the refund path stays in sight. The
 *      portal can cancel a subscription but cannot hand a payment back: a
 *      customer inside the window who only sees "manage" would cancel instead of
 *      being refunded, and lose money they were owed.
 */

const h = vi.hoisted(() => ({
  push: vi.fn(),
  replace: vi.fn(),
  refresh: vi.fn(),
  openPortal: vi.fn(),
  refundEligibility: vi.fn(),
  subscription: vi.fn(),
}));

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: h.push, replace: h.replace, refresh: h.refresh }),
  usePathname: () => '/compte',
}));

vi.mock('@/lib/auth/store', () => ({
  useAuth: () => ({
    account: { id: 1, username: 'me', email: 'me@x.io', role: 'user', consents: [] },
    loading: false,
    probeFailed: false,
    isOwner: false,
    logout: vi.fn(),
    refresh: vi.fn(),
  }),
}));

vi.mock('@/lib/i18n/href', () => ({
  useLocalizedHref: () => (path: string) => path,
}));

vi.mock('@/lib/billing/api-client', () => ({
  BillingError: class BillingError extends Error {
    status: number;
    constructor(status: number, message: string) {
      super(message);
      this.status = status;
      this.name = 'BillingError';
    }
  },
  openPortal: () => h.openPortal(),
  fetchRefundEligibility: () => h.refundEligibility(),
  fetchSubscription: () => h.subscription(),
}));

import { AccountPanel } from '../AccountPanel';
import { BillingError } from '@/lib/billing/api-client';

const NO_WINDOW = {
  eligible: false,
  reason: 'not_annual',
  days_remaining: 0,
  guarantee_days: 14,
  deadline: null,
};

const OPEN_WINDOW = {
  eligible: true,
  reason: null,
  days_remaining: 11,
  guarantee_days: 14,
  deadline: 1_800_000_000,
};

/** Rendered panel, with every query scoped to it rather than to the document. */
function renderPanel() {
  const { container } = render(<AccountPanel />);
  return within(container);
}

beforeEach(() => {
  h.push.mockReset();
  h.openPortal.mockReset();
  h.refundEligibility.mockReset().mockResolvedValue(NO_WINDOW);
  // Abonnement actif par défaut : la ligne « Plan actuel » de /compte le lit.
  h.subscription.mockReset().mockResolvedValue({
    status: 'active',
    price_id: 'price_m',
    current_period_end: null,
    cancel_at_period_end: false,
    trial_end: null,
    has_access: true,
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('LEG-2 — /compte opens the Stripe portal in one click', () => {
  it('mints a portal session on click instead of routing to /abonnement', async () => {
    h.openPortal.mockResolvedValue('https://billing.stripe.test/session/abc');
    const panel = renderPanel();

    fireEvent.click(panel.getByTestId('manage-subscription'));

    await waitFor(() => expect(h.openPortal).toHaveBeenCalledTimes(1));
    // The detour is gone: nothing navigated to our own subscription page.
    expect(h.push).not.toHaveBeenCalledWith('/abonnement');
  });

  it('sends an account with no billing yet to the plan choice, not to an error', async () => {
    h.openPortal.mockRejectedValue(new BillingError(409, 'No billing account yet'));
    const panel = renderPanel();

    fireEvent.click(panel.getByTestId('manage-subscription'));

    await waitFor(() => expect(h.push).toHaveBeenCalledWith('/abonnement'));
    expect(panel.queryByText(fr.app.account.manageError)).toBeNull();
  });

  it('says so when the portal cannot open, and keeps the Stripe login page', async () => {
    h.openPortal.mockRejectedValue(new BillingError(502, 'Stripe indisponible'));
    const panel = renderPanel();

    fireEvent.click(panel.getByTestId('manage-subscription'));

    // The server's own message, not a generic failure.
    expect(await panel.findByText('Stripe indisponible')).toBeInTheDocument();
    const link = panel.getByTestId('stripe-portal-link');
    expect(link).toHaveAttribute('href', STRIPE_PORTAL_LOGIN_URL);
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'));
  });

  it('keeps the refund path in sight while the 14-day window is open', async () => {
    h.refundEligibility.mockResolvedValue(OPEN_WINDOW);
    const panel = renderPanel();

    const cta = await panel.findByTestId('refund-window-cta');
    // /abonnement is where the guarantee is honoured in one click — the portal
    // cannot do it.
    expect(cta).toHaveAttribute('href', '/abonnement');
    expect(panel.getByTestId('refund-window-open')).toHaveTextContent('14');
  });

  it('shows no guarantee row when the window is closed', async () => {
    const panel = renderPanel();

    await waitFor(() => expect(h.refundEligibility).toHaveBeenCalled());
    expect(panel.queryByTestId('refund-window-open')).toBeNull();
  });
});
