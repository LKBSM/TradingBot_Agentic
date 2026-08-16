'use client';

import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import { googleFinalize } from '@/lib/auth/google';
import { useAuth } from '@/lib/auth/store';
import { useLocalizedHref } from '@/lib/i18n/href';
import { Button } from '@/components/ui/button';
import { CheckField, FormError, TextField } from './fields';

/**
 * Consent step that completes a Google signup (Loi 25 / 18+): Google verified
 * the email, but the account is only created after the user picks a username and
 * accepts 18+ / Conditions / Privacy — the same gates as email/password signup.
 */
export function GoogleFinalizeForm() {
  const t = useTranslations('auth');
  const params = useSearchParams();
  const token = params.get('tok') ?? '';
  const { refresh } = useAuth();
  const router = useRouter();
  const lh = useLocalizedHref();
  const [error, setError] = React.useState<string | null>(null);
  const [submitting, setSubmitting] = React.useState(false);
  const submittingRef = React.useRef(false);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (submittingRef.current) return;
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
      await googleFinalize({
        token,
        username: String(form.get('username') ?? '').trim(),
        age_confirmed: ageConfirmed,
        accept_terms: acceptTerms,
        accept_privacy: acceptPrivacy,
      });
      await refresh();
      router.refresh();
      // PAY-2 — a brand-new Google account is email-verified but has no
      // subscription yet: send it straight to the plan choice (paying is the
      // condition of entry). A subscriber sees their active state there.
      router.replace(lh('/abonnement'));
    } catch (err) {
      setError(err instanceof Error ? err.message : t('register.errorGeneric'));
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  }

  if (!token) {
    return <FormError message={t('google.missingToken')} />;
  }

  return (
    <form onSubmit={onSubmit} className="space-y-4" noValidate>
      <div>
        <h1 className="text-lg font-semibold">{t('google.finalizeTitle')}</h1>
        <p className="text-sm text-muted-foreground">{t('google.finalizeIntro')}</p>
      </div>
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
      <fieldset className="space-y-2.5 rounded-md border border-border/60 bg-muted/20 p-3">
        <legend className="px-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {t('register.declarationsLegend')}
        </legend>
        <CheckField name="age_confirmed" label={t('register.ageLabel')} />
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
        {submitting ? t('register.submitting') : t('google.finalizeSubmit')}
      </Button>
    </form>
  );
}
