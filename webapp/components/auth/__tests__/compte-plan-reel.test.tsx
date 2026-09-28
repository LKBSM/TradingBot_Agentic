import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import fr from '@/messages/fr.json';

/**
 * /compte — la ligne « Plan actuel » dit l'état RÉEL.
 *
 * Elle affichait « Aucune carte requise pendant l'accès anticipé », avec un badge
 * « Accès anticipé ». C'était faux depuis PAY-2, où payer est devenu la condition
 * d'entrée — et cette phrase se trouvait juste au-dessus du bouton qui ouvre le
 * portail de paiement.
 *
 * Sous un intitulé « Plan actuel », une phrase figée est une AFFIRMATION sur ce
 * que le client paie. Soit on dit l'état réel, soit on n'affirme rien. Ces tests
 * figent les trois cas : abonné, non abonné, et état inconnu — ce dernier étant
 * le piège, car c'est là qu'une interface est tentée d'inventer une valeur
 * rassurante.
 */

const h = vi.hoisted(() => ({
  subscription: vi.fn(),
  refundEligibility: vi.fn(),
  openPortal: vi.fn(),
  isOwner: { value: false },
}));

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  usePathname: () => '/compte',
}));

vi.mock('@/lib/auth/store', () => ({
  useAuth: () => ({
    account: { id: 1, username: 'me', email: 'me@x.io', role: 'user', consents: [] },
    loading: false,
    probeFailed: false,
    isOwner: h.isOwner.value,
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

const NO_WINDOW = {
  eligible: false,
  reason: null,
  days_remaining: 0,
  guarantee_days: 14,
  deadline: null,
};

function sub(over: Record<string, unknown> = {}) {
  return {
    status: 'active',
    price_id: 'price_m',
    current_period_end: null,
    cancel_at_period_end: false,
    trial_end: null,
    has_access: true,
    ...over,
  };
}

function mount() {
  return render(
    <NextIntlClientProvider locale="fr" messages={fr}>
      <AccountPanel />
    </NextIntlClientProvider>,
  );
}

beforeEach(() => {
  h.isOwner.value = false;
  h.openPortal.mockReset();
  h.refundEligibility.mockReset().mockResolvedValue(NO_WINDOW);
  h.subscription.mockReset().mockResolvedValue(sub());
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('/compte — « Plan actuel » ne ment plus', () => {
  it("n'affirme jamais qu'aucune carte n'est requise", async () => {
    mount();
    await waitFor(() => expect(screen.getByTestId('plan-value')).toBeInTheDocument());
    const page = document.body.textContent ?? '';
    // La phrase exacte, et la promesse qu'elle portait.
    expect(page).not.toContain('Aucune carte requise');
    expect(page).not.toMatch(/aucune carte/i);
  });

  it('la clé de copie fautive a disparu du fichier de langue', () => {
    // Retirée plutôt que laissée orpheline : une phrase morte finit par être
    // réutilisée par quelqu'un qui la croit vraie.
    const account = (fr as Record<string, any>).app.account;
    expect('planValue' in account).toBe(false);
    expect(JSON.stringify(account)).not.toMatch(/aucune carte/i);
  });

  it('un abonné voit son abonnement décrit comme actif', async () => {
    mount();
    await waitFor(() =>
      expect(screen.getByTestId('plan-badge')).toHaveTextContent(fr.billing.status.active),
    );
  });

  it('un compte SANS abonnement ne voit pas « actif »', async () => {
    h.subscription.mockResolvedValue(null);
    mount();
    await waitFor(() =>
      expect(screen.getByTestId('plan-badge')).toHaveTextContent(fr.billing.status.none),
    );
    expect(screen.getByTestId('plan-badge')).not.toHaveTextContent(fr.billing.status.active);
  });

  it('un abonnement en résiliation reste un accès actif', async () => {
    // `cancel_at_period_end` : le client garde l'accès jusqu'à la fin de la
    // période payée. Afficher « expiré » lui ferait croire qu'il a déjà perdu
    // ce qu'il a payé.
    h.subscription.mockResolvedValue(sub({ cancel_at_period_end: true }));
    mount();
    await waitFor(() =>
      expect(screen.getByTestId('plan-badge')).toHaveTextContent(fr.billing.status.active),
    );
  });

  it('un paiement échoué ne coupe pas encore l\'accès', async () => {
    h.subscription.mockResolvedValue(sub({ status: 'past_due' }));
    mount();
    await waitFor(() =>
      expect(screen.getByTestId('plan-badge')).toHaveTextContent(fr.billing.status.pastDue),
    );
  });

  it("quand l'appel échoue, la ligne le DIT au lieu d'inventer un plan", async () => {
    // Le cas qui compte : une interface qui ne sait pas doit se taire, pas
    // afficher une valeur rassurante. Et aucun badge d'état n'est affiché.
    h.subscription.mockRejectedValue(new Error('réseau'));
    mount();
    await waitFor(() => expect(screen.getByTestId('plan-value')).toBeInTheDocument());
    expect(screen.getByTestId('plan-value')).toHaveTextContent(fr.app.account.planUnknown);
    expect(screen.queryByTestId('plan-badge')).toBeNull();
  });

  it('un appel de facturation qui échoue ne casse pas le reste de la page', async () => {
    // /compte porte la session, le courriel et la déconnexion : rien de tout
    // cela ne doit dépendre d'un appel de facturation.
    h.subscription.mockRejectedValue(new Error('réseau'));
    h.refundEligibility.mockRejectedValue(new Error('réseau'));
    mount();
    await waitFor(() => expect(screen.getByText('me@x.io')).toBeInTheDocument());
  });

  it('le propriétaire est décrit comme tel, sans badge d\'abonnement', async () => {
    h.isOwner.value = true;
    mount();
    await waitFor(() => expect(screen.getByTestId('plan-value')).toBeInTheDocument());
    expect(screen.getByTestId('plan-value')).toHaveTextContent(fr.app.account.planOwner);
    expect(screen.queryByTestId('plan-badge')).toBeNull();
  });
});
