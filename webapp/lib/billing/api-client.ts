/**
 * Billing API client — talks to the account-bound FastAPI billing routes
 * through the same-origin `/api/*` rewrite. Auth is the first-party session
 * cookie (same as the auth client), so requests carry it automatically and no
 * token is exposed to JS. NO card data is ever handled here — Checkout and the
 * Customer Portal are hosted by Stripe; we only ever follow a returned URL.
 */

const BASE = '/api/billing';

export class BillingError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'BillingError';
  }
}

export interface Plan {
  key: string;
  price_id: string;
}

export interface Pricing {
  plans: Plan[];
  trial_days: number;
  tax_enabled: boolean;
}

export interface Subscription {
  status: string | null;
  price_id: string | null;
  current_period_end: number | null;
  cancel_at_period_end: boolean;
  trial_end: number | null;
  has_access: boolean;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { 'content-type': 'application/json' },
      ...init,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Erreur réseau';
    throw new BillingError(0, `Connexion impossible : ${message}`);
  }

  let parsed: unknown = null;
  const text = await res.text();
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = null;
    }
  }

  if (!res.ok) {
    const detail = (parsed as { detail?: unknown } | null)?.detail;
    const message =
      typeof detail === 'string'
        ? detail
        : 'Une erreur est survenue. Réessaie dans un instant.';
    throw new BillingError(res.status, message);
  }

  return parsed as T;
}

export function fetchPricing(): Promise<Pricing> {
  return request<Pricing>('/pricing', { method: 'GET' });
}

/** Current subscription, or null when not authenticated. */
export async function fetchSubscription(): Promise<Subscription | null> {
  try {
    return await request<Subscription>('/subscription', { method: 'GET' });
  } catch (err) {
    if (err instanceof BillingError && err.status === 401) return null;
    throw err;
  }
}

/** Start Checkout for a plan — resolves to the hosted Stripe URL to redirect to. */
export async function startCheckout(planKey: string): Promise<string> {
  const { url } = await request<{ url: string }>('/checkout', {
    method: 'POST',
    body: JSON.stringify({ plan_key: planKey }),
  });
  return url;
}

/** Open the Stripe Customer Portal — resolves to the hosted URL to redirect to. */
export async function openPortal(): Promise<string> {
  const { url } = await request<{ url: string }>('/portal', { method: 'POST' });
  return url;
}
