import * as React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { fireEvent } from '@testing-library/dom';
import { NextIntlClientProvider } from 'next-intl';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';
import es from '@/messages/es.json';

/**
 * LEG-1 — consent before payment.
 *
 * Nothing may reach Stripe Checkout until the customer has ticked a box that is
 * NOT pre-ticked, and the version they accepted must be recorded first. These
 * are the properties the mission asks for, asserted on the real component:
 *
 *  1. the box starts unticked;
 *  2. both plan CTAs are inactive while it is unticked, with the reason WRITTEN
 *     (a greyed button that says nothing is not an explanation);
 *  3. ticking activates them;
 *  4. paying records the consent BEFORE leaving for Stripe;
 *  5. if recording fails, the customer does NOT reach Checkout — we never take
 *     money with no trace of what was accepted;
 *  6. both documents are linked from this very screen.
 */

const startCheckout = vi.fn();
const acceptConsents = vi.fn();

vi.mock('@/lib/billing/api-client', () => ({
  // Declared INSIDE the factory: vi.mock is hoisted above module scope, so a
  // top-level class here would be read before initialisation.
  BillingError: class BillingError extends Error {},
  fetchPricing: async () => ({
    plans: [
      { key: 'MONTHLY', price_id: 'price_m' },
      { key: 'ANNUAL', price_id: 'price_a' },
    ],
  }),
  fetchSubscription: async () => null,
  fetchRefundEligibility: async () => ({
    eligible: false, reason: null, days_remaining: 0, guarantee_days: 14, deadline: null,
  }),
  requestRefund: async () => ({ refunded: true, amount: null, currency: null }),
  startCheckout: (...args: unknown[]) => startCheckout(...args),
  openPortal: async () => '',
  syncSubscription: async () => null,
}));

vi.mock('@/lib/auth/api-client', () => ({
  acceptConsents: (...args: unknown[]) => acceptConsents(...args),
  AuthError: class extends Error {},
}));

vi.mock('@/lib/auth/store', () => {
  // STABLE identity, deliberately. The panel's load effect depends on
  // `account`; returning a fresh object on every render (as a naive mock does)
  // re-runs the effect, which sets state, which renders again — an infinite
  // loop that hangs the test runner rather than failing it. In the app the
  // value comes from a context and is stable.
  const account = { id: 1, email: 'a@b.co', role: 'user' };
  const value = { account, loading: false };
  return { useAuth: () => value };
});

vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

import { SubscriptionPanel } from '../SubscriptionPanel';

const MESSAGES = { fr, en, es } as const;
type Loc = keyof typeof MESSAGES;
const LOCALES = ['fr', 'en', 'es'] as const;

function renderIn(locale: Loc) {
  return render(<SubscriptionPanel />, {
    wrapper: ({ children }) => (
      <NextIntlClientProvider locale={locale} messages={MESSAGES[locale]}>
        {children}
      </NextIntlClientProvider>
    ),
  });
}

/** The two plan CTAs — the only doors to Stripe on this screen. */
function planButtons(): HTMLButtonElement[] {
  return Array.from(
    document.querySelectorAll<HTMLButtonElement>('.grid button'),
  );
}

function consentBox(): HTMLInputElement {
  return screen.getByTestId('consent-checkbox') as HTMLInputElement;
}

beforeEach(() => {
  startCheckout.mockReset().mockResolvedValue('https://checkout.stripe.test/s/1');
  acceptConsents.mockReset().mockResolvedValue({ consents: [] });
});

describe.each(LOCALES)('LEG-1 consent gate [%s]', (locale) => {
  it('the box is NOT pre-ticked', async () => {
    renderIn(locale);
    await waitFor(() => expect(planButtons()).toHaveLength(2));
    expect(consentBox().checked).toBe(false);
  });

  it('both plan CTAs are inactive while the box is unticked', async () => {
    renderIn(locale);
    await waitFor(() => expect(planButtons()).toHaveLength(2));
    for (const button of planButtons()) expect(button).toBeDisabled();
  });

  it('the reason is written out, and tied to the disabled buttons', async () => {
    renderIn(locale);
    await waitFor(() => expect(planButtons()).toHaveLength(2));
    const reason = screen.getByTestId('consent-blocked');
    expect(reason.textContent).toBe(MESSAGES[locale].billing.consent.blocked);
    for (const button of planButtons()) {
      expect(button.getAttribute('aria-describedby')).toBe(reason.id);
    }
  });

  it('ticking the box activates both CTAs and drops the reason', async () => {
    renderIn(locale);
    await waitFor(() => expect(planButtons()).toHaveLength(2));
    fireEvent.click(consentBox());
    await waitFor(() => {
      for (const button of planButtons()) expect(button).toBeEnabled();
    });
    expect(screen.queryByTestId('consent-blocked')).toBeNull();
  });

  it('links to BOTH documents from this screen', async () => {
    renderIn(locale);
    await waitFor(() => expect(planButtons()).toHaveLength(2));
    const hrefs = Array.from(document.querySelectorAll('a')).map((a) =>
      a.getAttribute('href'),
    );
    expect(hrefs.some((h) => h?.includes('/conditions'))).toBe(true);
    expect(hrefs.some((h) => h?.includes('/confidentialite'))).toBe(true);
  });
});

describe('LEG-1 consent gate — what happens on payment', () => {
  it('records the consent BEFORE starting checkout', async () => {
    const order: string[] = [];
    acceptConsents.mockImplementation(async () => {
      order.push('consent');
      return { consents: [] };
    });
    startCheckout.mockImplementation(async () => {
      order.push('checkout');
      return 'https://checkout.stripe.test/s/1';
    });

    renderIn('fr');
    await waitFor(() => expect(planButtons()).toHaveLength(2));
    fireEvent.click(consentBox());
    await waitFor(() => expect(planButtons()[0]).toBeEnabled());
    fireEvent.click(planButtons()[0]!);

    await waitFor(() => expect(order).toEqual(['consent', 'checkout']));
  });

  it('does NOT reach checkout when recording the consent fails', async () => {
    acceptConsents.mockRejectedValue(new Error('offline'));

    renderIn('fr');
    await waitFor(() => expect(planButtons()).toHaveLength(2));
    fireEvent.click(consentBox());
    await waitFor(() => expect(planButtons()[0]).toBeEnabled());
    fireEvent.click(planButtons()[0]!);

    await waitFor(() =>
      expect(screen.getByRole('alert').textContent).toBe(
        fr.billing.consent.error,
      ),
    );
    expect(startCheckout).not.toHaveBeenCalled();
  });

  it('a programmatic click without consent still reaches nothing', async () => {
    // Defence in depth: the guard is not only the `disabled` attribute.
    renderIn('fr');
    await waitFor(() => expect(planButtons()).toHaveLength(2));
    const button = planButtons()[0]!;
    button.removeAttribute('disabled');
    fireEvent.click(button);
    await new Promise((r) => setTimeout(r, 0));
    expect(acceptConsents).not.toHaveBeenCalled();
    expect(startCheckout).not.toHaveBeenCalled();
  });
});
