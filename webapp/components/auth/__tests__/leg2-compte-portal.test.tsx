import { render, screen } from '@/components/test-utils';
import { describe, expect, it, vi } from 'vitest';
import fr from '@/messages/fr.json';
import { STRIPE_PORTAL_LOGIN_URL } from '@/lib/billing/portal';

/**
 * LEG-2 — where /compte sends someone who wants to cancel.
 *
 * Clause 8.1 of the terms now tells the customer that cancelling happens in
 * their customer area, through « Gérer mon abonnement », which opens the Stripe
 * portal. That sentence is a contractual promise, so the two doors it depends on
 * are held here:
 *
 *   « Gérer l'abonnement » → /abonnement, where the « Gérer mon abonnement »
 *   button mints a portal session (see components/billing/__tests__);
 *   « Portail client Stripe » → Stripe's own login page, which keeps cancelling
 *   possible even when our backend is down.
 */

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  usePathname: () => '/compte',
}));

vi.mock('@/lib/auth/store', () => ({
  useAuth: () => ({
    account: { id: 1, username: 'me', email: 'me@x.io', role: 'user', consents: [] },
    loading: false,
    probeFailed: false,
    logout: vi.fn(),
    refresh: vi.fn(),
  }),
}));

vi.mock('@/lib/i18n/href', () => ({
  useLocalizedHref: () => (path: string) => path,
}));

import { AccountPanel } from '../AccountPanel';

describe('LEG-2 — /compte keeps a path to cancelling', () => {
  it('sends « Gérer l\'abonnement » to the subscription page', () => {
    render(<AccountPanel />);

    const row = screen.getByText(fr.app.account.manageRow).closest('.setrow');
    expect(row).not.toBeNull();
    const cta = row!.querySelector('a');
    expect(cta).toHaveAttribute('href', '/abonnement');
    expect(cta).toHaveTextContent(fr.app.account.manageCta);
  });

  it('keeps the Stripe portal login page reachable directly', () => {
    render(<AccountPanel />);

    const link = screen.getByTestId('stripe-portal-link');
    expect(link).toHaveAttribute('href', STRIPE_PORTAL_LOGIN_URL);
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'));
  });
});
