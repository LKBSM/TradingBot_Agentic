'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import { AuthError } from '@/lib/auth/api-client';
import { useAuth } from '@/lib/auth/store';
import { useLocalizedHref } from '@/lib/i18n/href';
import { Button } from '@/components/ui/button';
import { CheckField, FormError, TextField } from './fields';

/**
 * Registration form. Enforces, client-side, the same gates the backend enforces
 * server-side (defence in depth, never the only check): 18+ self-declaration +
 * explicit acceptance of the Conditions and the Privacy Policy. The version and
 * timestamp of those consents are recorded server-side at account creation.
 */
export function RegisterForm() {
  const t = useTranslations('auth');
  const { register } = useAuth();
  const router = useRouter();
  const lh = useLocalizedHref();
  const [error, setError] = React.useState<string | null>(null);
  const [submitting, setSubmitting] = React.useState(false);
  const submittingRef = React.useRef(false);

  // Closed private beta: the backend 403s every /register call. Don't offer a
  // form the server is guaranteed to reject — show a clear "registrations
  // closed" notice instead (AUTH-18). Same public flag that drives the gate.
  const registrationsClosed = process.env.NEXT_PUBLIC_BETA_LOCKDOWN === '1';

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (submittingRef.current) return; // AUTH-08 — no double submit
    setError(null);
    const form = new FormData(e.currentTarget);
    const ageConfirmed = form.get('age_confirmed') === 'on';
    const acceptTerms = form.get('accept_terms') === 'on';
    const acceptPrivacy = form.get('accept_privacy') === 'on';

    if (!ageConfirmed) {
      setError(t('register.errorAge'));
      return;
    }
    if (!acceptTerms || !acceptPrivacy) {
      setError(t('register.errorConsents'));
      return;
    }

    submittingRef.current = true;
    setSubmitting(true);
    try {
      await register({
        username: String(form.get('username') ?? '').trim(),
        email: String(form.get('email') ?? '').trim(),
        password: String(form.get('password') ?? ''),
        age_confirmed: ageConfirmed,
        accept_terms: acceptTerms,
        accept_privacy: acceptPrivacy,
      });
      // Registration also opens a session (Set-Cookie). Same first-attempt
      // reliability fix as login: invalidate the Router Cache before navigating
      // so the destination is fetched fresh with the new cookie. See LoginForm.
      router.refresh();
      router.replace(lh('/compte'));
    } catch (err) {
      setError(
        err instanceof AuthError ? err.message : t('register.errorGeneric'),
      );
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  }

  if (registrationsClosed) {
    return (
      <div className="space-y-3 rounded-lg border border-border/60 bg-muted/20 p-5 text-center">
        <h2 className="text-base font-medium text-foreground">
          {t('register.closedTitle')}
        </h2>
        <p className="text-sm text-muted-foreground">{t('register.closedBody')}</p>
        <p className="text-sm text-muted-foreground">
          {t('register.haveAccount')}{' '}
          <Link href={lh('/connexion')} className="underline underline-offset-2 hover:text-foreground">
            {t('register.login')}
          </Link>
        </p>
      </div>
    );
  }

  return (
    <form onSubmit={onSubmit} className="space-y-4" noValidate>
      <FormError message={error} />
      <TextField
        label={t('register.usernameLabel')}
        name="username"
        autoComplete="username"
        required
        minLength={3}
        maxLength={32}
        hint={t('register.usernameHint')}
      />
      <TextField
        label={t('register.emailLabel')}
        name="email"
        type="email"
        autoComplete="email"
        required
      />
      <TextField
        label={t('register.passwordLabel')}
        name="password"
        type="password"
        autoComplete="new-password"
        required
        minLength={10}
        hint={t('register.passwordHint')}
      />

      <fieldset className="space-y-2.5 rounded-md border border-border/60 bg-muted/20 p-3">
        <legend className="px-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {t('register.declarationsLegend')}
        </legend>
        <CheckField
          name="age_confirmed"
          label={t('register.ageLabel')}
        />
        <CheckField
          name="accept_terms"
          label={t.rich('register.acceptTerms', {
            link: (chunks) => (
              <Link href={lh('/conditions')} className="underline underline-offset-2 hover:text-foreground">
                {chunks}
              </Link>
            ),
          })}
        />
        <CheckField
          name="accept_privacy"
          label={t.rich('register.acceptPrivacy', {
            link: (chunks) => (
              <Link href={lh('/confidentialite')} className="underline underline-offset-2 hover:text-foreground">
                {chunks}
              </Link>
            ),
          })}
        />
      </fieldset>

      <Button type="submit" className="w-full" disabled={submitting}>
        {submitting ? t('register.submitting') : t('register.submit')}
      </Button>

      <p className="text-center text-sm text-muted-foreground">
        {t('register.haveAccount')}{' '}
        <Link href={lh('/connexion')} className="underline underline-offset-2 hover:text-foreground">
          {t('register.login')}
        </Link>
      </p>
    </form>
  );
}
