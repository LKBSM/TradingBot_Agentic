'use client';

import Link from 'next/link';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import { AuthError, requestPasswordReset } from '@/lib/auth/api-client';
import { useLocalizedHref } from '@/lib/i18n/href';
import { Button } from '@/components/ui/button';
import { FormError, FormSuccess, TextField } from './fields';

/**
 * Password-reset request. The backend answers identically whether or not the
 * identifier matched (anti-enumeration), so this form always shows the same
 * neutral confirmation on success.
 */
export function ForgotPasswordForm() {
  const t = useTranslations('auth');
  const lh = useLocalizedHref();
  const [error, setError] = React.useState<string | null>(null);
  const [done, setDone] = React.useState<string | null>(null);
  const [submitting, setSubmitting] = React.useState(false);
  const submittingRef = React.useRef(false);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (submittingRef.current) return; // AUTH-08 — no double submit
    submittingRef.current = true;
    setError(null);
    const form = new FormData(e.currentTarget);
    setSubmitting(true);
    try {
      const res = await requestPasswordReset(
        String(form.get('identifier') ?? '').trim(),
      );
      setDone(res.message);
    } catch (err) {
      setError(err instanceof AuthError ? err.message : t('forgot.errorGeneric'));
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  }

  if (done) {
    return (
      <div className="space-y-4">
        <FormSuccess message={done} />
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
      <p className="text-sm text-muted-foreground">
        {t('forgot.intro')}
      </p>
      <TextField
        label={t('forgot.identifierLabel')}
        name="identifier"
        autoComplete="username"
        required
      />
      <Button type="submit" className="w-full" disabled={submitting}>
        {submitting ? t('forgot.submitting') : t('forgot.submit')}
      </Button>
      <p className="text-center text-sm text-muted-foreground">
        <Link href={lh('/connexion')} className="underline underline-offset-2 hover:text-foreground">
          {t('forgot.backToLogin')}
        </Link>
      </p>
    </form>
  );
}
