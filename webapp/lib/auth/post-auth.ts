/**
 * Post-authentication routing (PAY-3) — the single place that maps a freshly
 * authenticated account to the correct one of the six states. Both sign-in paths
 * (email/password and Google) route through this so a returning account never
 * lands on the product shell just to hit a paywall a beat later.
 *
 * The decision mirrors the server gate order exactly:
 *   1. gate OFF (testing) or has access → the product (honor a safe ?next=).
 *   2. email not verified               → the "confirm your email" screen.
 *   3. authenticated, verified, no sub  → the plan-choice / subscribe screen.
 *
 * It never throws: the access probe is a DISPLAY hint, and the server-side guard
 * remains the real wall — so on any probe failure we fall back to the product.
 */

import { fetchAccess } from '@/lib/access/api-client';

export interface PostAuthTargets {
  /** Localized href to the product (/app). */
  appHref: string;
  /** Localized href to the plan-choice page (/abonnement). */
  subscribeHref: string;
  /** Localized href to the email-confirmation screen (/verifier-email). */
  verifyHref: string;
  /** A safe, same-site internal path from ?next=, honored only WITH access. */
  next?: string | null;
}

export async function resolvePostAuthDestination(
  t: PostAuthTargets,
): Promise<string> {
  try {
    const a = await fetchAccess();
    // Full access (or the gate is off during testing) → the product. A ?next=
    // return path is honored ONLY here — routing an un-entitled account to a
    // gated `next` would just bounce it straight back.
    if (!a.gate_enforced || a.has_access) return t.next || t.appHref;
    // Verification wall comes before the subscribe wall.
    if (a.email_verification_required) return t.verifyHref;
    // Authenticated + verified, but no active subscription → choose a plan.
    return t.subscribeHref;
  } catch {
    return t.next || t.appHref;
  }
}
