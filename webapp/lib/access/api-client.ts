/**
 * Access summary client — reads GET /api/access/me to learn what the current
 * account may see. Same-origin (the session cookie rides along automatically via
 * the `/api/*` rewrite). This drives the DISPLAY (lock vs. show); the server-side
 * guards remain the non-bypassable source of truth.
 */

export interface AccessSummary {
  authenticated: boolean;
  /** False during the personal-testing phase — everything is then open. */
  gate_enforced: boolean;
  /** Closed-beta login wall. When true the whole product API is 401 for anon. */
  beta_lockdown: boolean;
  /** Convenience: `beta_lockdown && !authenticated` — the UI must route to login. */
  must_login: boolean;
  is_owner: boolean;
  /**
   * PAY-1 is paid-only — access is all-or-nothing. True when the gate is off
   * (testing) or the account has an active subscription (or is the owner).
   */
  has_access: boolean;
  /**
   * Authenticated but not entitled — the "account page + subscribe invitation"
   * state. Drives the paywall/upsell.
   */
  subscription_required: boolean;
}

const ENDPOINT = '/api/access/me';

// REC-1: a request without a response must never leave the UI in a perpetual
// loading state. The SubscriptionGate wraps EVERY product page and spins while
// this call is unresolved; with no timeout, a slow/blocked backend left every
// page stuck on the skeleton with no error. This bounds the wait so the gate
// always reaches a decision (fail-open to children off-lockdown, per its
// existing design) instead of spinning forever.
const ACCESS_TIMEOUT_MS = 8_000;

/** Fetch the caller's access summary. Never throws on 401 (returns the payload,
 * which says `authenticated:false`); throws only on transport/parse/timeout
 * failure. Bounded by ACCESS_TIMEOUT_MS so the gate can never spin forever. */
export async function fetchAccess(signal?: AbortSignal): Promise<AccessSummary> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ACCESS_TIMEOUT_MS);
  const onCallerAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener('abort', onCallerAbort, { once: true });
  }
  let res: Response;
  try {
    res = await fetch(ENDPOINT, {
      method: 'GET',
      headers: { accept: 'application/json' },
      // Explicitly send the same-origin HttpOnly session cookie. `same-origin` is
      // the fetch default, but we set it for parity with the auth client and to
      // keep the gate decision working if these calls are ever proxied — the whole
      // access decision depends on the session cookie riding along.
      credentials: 'same-origin',
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onCallerAbort);
  }
  if (!res.ok) {
    throw new Error(`access summary unavailable (${res.status})`);
  }
  return (await res.json()) as AccessSummary;
}
