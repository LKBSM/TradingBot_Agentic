'use client';

import Link from 'next/link';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import { AuthError, confirmPasswordReset } from '@/lib/auth/api-client';
import { useLocalizedHref } from '@/lib/i18n/href';
import { Button } from '@/components/ui/button';
import { FormError, FormSuccess, TextField } from './fields';

/**
 * Password-reset confirmation (AUTH-02). Reads the single-use `?token=` from the
 * URL (the link delivered by email) and lets the user set a new password via
 * `confirmPasswordReset`. Without a token it shows an honest error instead of a
 * dead form. Token read from `window.location` at submit time to avoid the
 * useSearchParams Suspense-boundary requirement on this route's build.
 */
export function ResetPasswordForm() {
  const t = useTranslations('auth');
  const lh = useLocalizedHref();
  const [error, setError] = React.useState<string | null>(null);
  const [done, setDone] = React.useState(false);
  const [submitting, setSubmitting] = React.useState(false);
  const submittingRef = React.useRef(false);

  function readToken(): string | null {
    if (typeof window === 'undefined') return null;
    return new URLSearchParams(window.location.search).get('token');
  }

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (submittingRef.current) return; // no double submit
    const token = readToken();
    if (!token) {
      setError(t('resetConfirm.missingToken'));
      return;
    }
    submittingRef.current = true;
    setError(null);
    const form = new FormData(e.currentTarget);
    setSubmitting(true);
    try {
      await confirmPasswordReset(token, String(form.get('password') ?? ''));
      setDone(true);
    } catch (err) {
      setError(
        err instanceof AuthError ? err.message : t('resetConfirm.errorGeneric'),
      );
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  }

  if (done) {
    return (
      <div className="space-y-4">
        <FormSuccess message={t('resetConfirm.success')} />
        <Link
          href={lh('/connexion')}
          className="block text-center text-sm text-muted-foreground underline underline-offset-2 hover:text-foreground"
        >
          {t('forgot.backToLogin')}
        </Link>
      </div>
    );
  }

  return (
    <form onSubmit={onSubmit} className="space-y-4" noValidate>
      <FormError message={error} />
      <p className="text-sm text-muted-foreground">{t('resetConfirm.intro')}</p>
      <TextField
        label={t('resetConfirm.passwordLabel')}
        name="password"
        type="password"
        autoComplete="new-password"
        required
        minLength={10}
      />
      <Button type="submit" className="w-full" disabled={submitting}>
        {submitting ? t('resetConfirm.submitting') : t('resetConfirm.submit')}
      </Button>
      <p className="text-center text-sm text-muted-foreground">
        <Link href={lh('/connexion')} className="underline underline-offset-2 hover:text-foreground">
          {t('forgot.backToLogin')}
        </Link>
      </p>
    </form>
  );
}
