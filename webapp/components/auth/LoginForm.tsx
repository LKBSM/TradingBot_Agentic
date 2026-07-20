'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import { AuthError } from '@/lib/auth/api-client';
import { useAuth } from '@/lib/auth/store';
import { useLocalizedHref } from '@/lib/i18n/href';
import { Button } from '@/components/ui/button';
import { FormError, TextField } from './fields';

/** Login form — identifier is a username OR an email (single field). */
export function LoginForm() {
  const t = useTranslations('auth');
  const { login } = useAuth();
  const router = useRouter();
  const lh = useLocalizedHref();
  const [error, setError] = React.useState<string | null>(null);
  const [submitting, setSubmitting] = React.useState(false);
  const submittingRef = React.useRef(false);

  // After login, go straight to the product. Honor a ?next= return path (set by
  // the login-wall redirect) when it's a safe internal path, else land on /app.
  // Read from window at submit time (client-only) to avoid the useSearchParams
  // Suspense-boundary requirement on the /connexion page build.
  function resolveDestination(): string {
    // `next` (set by the login wall) already carries the locale prefix; the
    // fallback is localized so a non-default-locale user lands on /<locale>/app
    // rather than the default locale (NAV-12).
    if (typeof window === 'undefined') return lh('/app');
    const next = new URLSearchParams(window.location.search).get('next');
    // Only a same-site absolute path is allowed. `//host` and `/\host` are
    // protocol-relative URLs the browser resolves off-site → open-redirect
    // (AUTH-06). Require a single leading slash not followed by / or \.
    if (next && /^\/(?![/\\])/.test(next)) return next;
    return lh('/app');
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
      const dest = resolveDestination();
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
    <form onSubmit={onSubmit} className="space-y-4" noValidate>
      <FormError message={error} />
      <TextField
        label={t('login.identifierLabel')}
        name="identifier"
        autoComplete="username"
        required
      />
      <TextField
        label={t('login.passwordLabel')}
        name="password"
        type="password"
        autoComplete="current-password"
        required
      />
      <div className="text-right">
        <Link
          href={lh('/mot-de-passe-oublie')}
          className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
        >
          {t('login.forgotPassword')}
        </Link>
      </div>
      <Button type="submit" className="w-full" disabled={submitting}>
        {submitting ? t('login.submitting') : t('login.submit')}
      </Button>
      <p className="text-center text-sm text-muted-foreground">
        {t('login.noAccount')}{' '}
        <Link href={lh('/inscription')} className="underline underline-offset-2 hover:text-foreground">
          {t('login.createAccount')}
        </Link>
      </p>
    </form>
  );
}
