'use client';

import * as React from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { fetchAccess, type AccessSummary } from '@/lib/access/api-client';
import { Paywall } from './Paywall';

export interface SubscriptionGateProps {
  /**
   * When true the WHOLE subtree requires full (paid/owner) access — used for
   * paid-only surfaces like the scanner. When false (default) an authenticated
   * free account is let through (it has a partial perimeter, e.g. XAU/USD M15);
   * locked combos then degrade per-request to a clean upsell.
   */
  requireFullAccess?: boolean;
  /** Copy shown on the paywall when access is insufficient. */
  paywallTitle?: string;
  paywallDescription?: string;
  children: React.ReactNode;
}

/** Derive the locale prefix ("" for the default fr, "/en" for en) from the path. */
function localePrefix(pathname: string): string {
  return pathname === '/en' || pathname.startsWith('/en/') ? '/en' : '';
}

/**
 * Client route guard for the gated product surfaces (`/app`, `/scanner`).
 *
 * Behaviour mirrors the server gate exactly:
 *   · gate OFF (testing phase)        → always renders children (open).
 *   · gate ON + not authenticated     → redirect to /connexion?next=…
 *   · gate ON + requireFullAccess + free → render <Paywall> (upsell).
 *   · otherwise                       → render children.
 *
 * It reads /api/access/me once; while loading it shows a minimal skeleton so the
 * page never flashes gated content before the decision is made.
 */
export function SubscriptionGate({
  requireFullAccess = false,
  paywallTitle,
  paywallDescription,
  children,
}: SubscriptionGateProps) {
  const router = useRouter();
  const pathname = usePathname() || '/';
  const [access, setAccess] = React.useState<AccessSummary | null>(null);
  const [error, setError] = React.useState(false);

  React.useEffect(() => {
    const controller = new AbortController();
    let active = true;
    fetchAccess(controller.signal)
      .then((a) => {
        if (active) setAccess(a);
      })
      .catch(() => {
        if (active) setError(true);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, []);

  const prefix = localePrefix(pathname);

  // Redirect unauthenticated users away from a gated page (effect, not render).
  // Two independent triggers:
  //   · beta lockdown (closed beta)  → must_login when not authenticated;
  //   · freemium/payment gate ON     → anonymous callers must log in.
  const mustLogin =
    access?.must_login === true ||
    (access?.gate_enforced === true && access.authenticated === false);
  React.useEffect(() => {
    if (!mustLogin) return;
    const next = encodeURIComponent(pathname);
    router.replace(`${prefix}/connexion?next=${next}`);
  }, [mustLogin, pathname, prefix, router]);

  // Transport failure handling:
  //   · closed beta (NEXT_PUBLIC_BETA_LOCKDOWN=1) → fail CLOSED: we cannot
  //     confirm a valid session, and every product API call is 401 anyway, so
  //     bounce to login rather than render a broken/empty shell.
  //   · otherwise → fail OPEN: the server guard is the real wall, so a flaky
  //     summary fetch must never hard-block a paying user during testing.
  const lockdown = process.env.NEXT_PUBLIC_BETA_LOCKDOWN === '1';
  React.useEffect(() => {
    if (!error || !lockdown) return;
    const next = encodeURIComponent(pathname);
    router.replace(`${prefix}/connexion?next=${next}`);
  }, [error, lockdown, pathname, prefix, router]);
  if (error) {
    if (lockdown) return null; // redirecting to login
    return <>{children}</>;
  }

  if (access === null || mustLogin) {
    return (
      <div
        className="flex min-h-[40vh] items-center justify-center"
        aria-busy="true"
        aria-live="polite"
      >
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-muted-foreground/30 border-t-primary" />
        <span className="sr-only">Chargement…</span>
      </div>
    );
  }

  const blocked =
    requireFullAccess && access.gate_enforced && !access.has_full_access;
  if (blocked) {
    return (
      <div className="container-wide py-12">
        <Paywall
          title={paywallTitle}
          description={paywallDescription}
          basePrefix={prefix}
        />
      </div>
    );
  }

  return <>{children}</>;
}
