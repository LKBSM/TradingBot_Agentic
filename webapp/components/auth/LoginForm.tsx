'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import { AuthError } from '@/lib/auth/api-client';
import { resolvePostAuthDestination } from '@/lib/auth/post-auth';
import { useAuth } from '@/lib/auth/store';
import { useLocalizedHref } from '@/lib/i18n/href';
import { BRAND_NAME } from '@/lib/brand';

/** Login form — identifier is a username OR an email (single field). */
export function LoginForm() {
  const t = useTranslations('auth');
  const tf = useTranslations('footer');
  const tc = useTranslations('connexion');
  const { login } = useAuth();
  const router = useRouter();
  const lh = useLocalizedHref();
  const [error, setError] = React.useState<string | null>(null);
  const [submitting, setSubmitting] = React.useState(false);
  const submittingRef = React.useRef(false);

  // A safe, same-site ?next= return path (set by the login wall), honored only
  // once the account has access (post-auth routing owns the six-state decision).
  function safeNext(): string | null {
    if (typeof window === 'undefined') return null;
    const next = new URLSearchParams(window.location.search).get('next');
    // Only a same-site absolute path is allowed. `//host` and `/\host` are
    // protocol-relative URLs the browser resolves off-site → open-redirect
    // (AUTH-06). Require a single leading slash not followed by / or \.
    return next && /^\/(?![/\\])/.test(next) ? next : null;
  }

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (submittingRef.current) return; // AUTH-08 — no double submit
    submittingRef.current = true;
    setError(null);
    const form = new FormData(e.currentTarget);
    setSubmitting(true);
    try {
      await login({
        identifier: String(form.get('identifier') ?? '').trim(),
        password: String(form.get('password') ?? ''),
      });
      // PAY-3 — route by the six states, not straight to /app: a returning
      // account with no active subscription goes to /abonnement (or the verify
      // screen), never to the product shell just to hit a paywall a beat later.
      const dest = await resolvePostAuthDestination({
        appHref: lh('/app'),
        subscribeHref: lh('/abonnement'),
        verifyHref: lh('/verifier-email'),
        next: safeNext(),
      });
      // The session cookie is now set. Invalidate the Next.js Router Cache
      // BEFORE navigating: during the cookieless visit the edge middleware
      // (BETA_LOCKDOWN) redirected the protected route to /connexion, and that
      // redirect can be cached client-side. Navigating without clearing it can
      // serve the stale "→ /connexion" entry and bounce a freshly-authenticated
      // user right back to login on the FIRST attempt — only a hard refresh
      // (which drops the Router Cache) then lets them in. `refresh()` first so
      // the destination is refetched from the server with the new cookie;
      // `replace` (not `push`) keeps /connexion out of history. This is what
      // makes login reliable on the first attempt.
      router.refresh();
      router.replace(dest);
    } catch (err) {
      setError(
        err instanceof AuthError ? err.message : t('login.errorGeneric'),
      );
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  }

  return (
    <div className="login-card">
      <div className="brandline">
        <div className="mark" aria-hidden="true">
          M
        </div>
        <b>{BRAND_NAME}</b>
        <span className="bl-sp" />
        <span className="ea">
          <span className="d" aria-hidden="true" />
          {tf('earlyAccessBadge')}
        </span>
      </div>
      <h1>{tc('title')}</h1>
      <p className="lsub">{tc('subtitle')}</p>

      <form onSubmit={onSubmit} noValidate>
        {error && (
          <p role="alert" className="form-error">
            {error}
          </p>
        )}
        <div className="fgroup">
          <label className="flabel" htmlFor="identifier">
            {t('login.identifierLabel')}
          </label>
          <input
            className="input"
            id="identifier"
            name="identifier"
            autoComplete="username"
            required
          />
        </div>
        <div className="fgroup">
          <label className="flabel" htmlFor="password">
            {t('login.passwordLabel')}
          </label>
          <input
            className="input"
            id="password"
            name="password"
            type="password"
            autoComplete="current-password"
            required
          />
        </div>
        <div className="frow">
          <Link href={lh('/mot-de-passe-oublie')}>
            {t('login.forgotPassword')}
          </Link>
        </div>
        <button type="submit" className="btn primary" disabled={submitting}>
          {submitting ? t('login.submitting') : t('login.submit')}
        </button>
      </form>

      <p className="mkacct">
        {t('login.noAccount')}{' '}
        <Link href={lh('/inscription')}>{t('login.createAccount')}</Link>
      </p>
      <div className="trust">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 3l8 4v5c0 4-3.4 7.4-8 9-4.6-1.6-8-5-8-9V7l8-4z" />
        </svg>
        <p>{tc('trust')}</p>
      </div>
    </div>
  );
}
