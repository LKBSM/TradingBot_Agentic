'use client';

import { useRouter, useSearchParams } from 'next/navigation';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import { MailCheck } from 'lucide-react';
import {
  confirmEmailVerification,
  confirmEmailVerificationCode,
  resendVerification,
} from '@/lib/auth/api-client';
import { useAuth } from '@/lib/auth/store';
import { useLocalizedHref } from '@/lib/i18n/href';
import { Button } from '@/components/ui/button';
import { FormError, FormSuccess } from './fields';

/**
 * Email-verification screen (PAY-3c), two modes on one page:
 *
 *  · WITH a ?token= (from the emailed link) → confirm it once, then forward to
 *    the plan-choice page.
 *  · WITHOUT a token (landed here right after signing up) → a clean card where
 *    the user TYPES the 6-digit code we emailed (primary path, no context
 *    switch), with a resend fallback.
 */
export function EmailVerifier() {
  const t = useTranslations('auth.verifyEmail');
  const params = useSearchParams();
  const token = params.get('token') ?? '';
  const { account, refresh } = useAuth();
  const router = useRouter();
  const lh = useLocalizedHref();

  const [state, setState] = React.useState<'verifying' | 'ok' | 'error'>(
    token ? 'verifying' : 'ok',
  );
  const ranRef = React.useRef(false);

  // Code entry (inbox mode).
  const [code, setCode] = React.useState('');
  const [checking, setChecking] = React.useState(false);
  const [codeError, setCodeError] = React.useState<string | null>(null);
  const [resendState, setResendState] = React.useState<
    'idle' | 'sending' | 'sent' | 'error'
  >('idle');

  React.useEffect(() => {
    if (!token) return;
    if (ranRef.current) return; // confirm exactly once (token is single-use)
    ranRef.current = true;
    confirmEmailVerification(token)
      .then(() => {
        setState('ok');
        void refresh();
      })
      .catch(() => setState('error'));
  }, [token, refresh]);

  React.useEffect(() => {
    if (!token || state !== 'ok') return;
    const id = setTimeout(() => router.push(lh('/abonnement')), 1200);
    return () => clearTimeout(id);
  }, [token, state, router, lh]);

  async function goToPlans() {
    await refresh();
    router.push(lh('/abonnement'));
  }

  async function onVerifyCode(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (checking) return;
    setCodeError(null);
    setChecking(true);
    try {
      await confirmEmailVerificationCode(code.trim());
      await goToPlans();
    } catch {
      setCodeError(t('codeError'));
      setChecking(false);
    }
  }

  async function onResend() {
    setResendState('sending');
    try {
      await resendVerification();
      setResendState('sent');
    } catch {
      setResendState('error');
    }
  }

  // ── Token mode (link clicked) ───────────────────────────────────────────
  if (token) {
    return (
      <div className="mx-auto max-w-md py-4">
        <div className="space-y-5 rounded-2xl border border-border/60 p-6 text-center sm:p-8">
          <h1 className="text-xl font-semibold tracking-tight">{t('title')}</h1>
          {state === 'verifying' && (
            <p className="text-sm text-muted-foreground" aria-live="polite">
              {t('verifying')}
            </p>
          )}
          {state === 'ok' && (
            <div className="space-y-4">
              <FormSuccess message={t('success')} />
              <Button className="w-full" onClick={goToPlans}>
                {t('cta')}
              </Button>
            </div>
          )}
          {state === 'error' && <FormError message={t('error')} />}
        </div>
      </div>
    );
  }

  // ── Inbox mode (just signed up) — type the 6-digit code ─────────────────
  return (
    <div className="mx-auto max-w-md py-4">
      <div className="space-y-6 rounded-2xl border border-border/60 p-6 sm:p-8">
        <div className="space-y-2 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
            <MailCheck className="h-6 w-6" aria-hidden />
          </div>
          <h1 className="text-xl font-semibold tracking-tight">{t('inboxTitle')}</h1>
          <p className="text-sm text-muted-foreground">
            {account?.email
              ? t('codeIntro', { email: account.email })
              : t('codeIntroNoEmail')}
          </p>
        </div>

        {codeError && <FormError message={codeError} />}
        {resendState === 'sent' && <FormSuccess message={t('resent')} />}
        {resendState === 'error' && <FormError message={t('resendError')} />}

        <form onSubmit={onVerifyCode} className="space-y-4">
          <div className="space-y-1.5">
            <label htmlFor="code" className="block text-sm font-medium text-foreground">
              {t('codeLabel')}
            </label>
            <input
              id="code"
              name="code"
              inputMode="numeric"
              autoComplete="one-time-code"
              autoFocus
              maxLength={6}
              value={code}
              onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
              className="w-full rounded-md border border-input bg-background px-3 py-3 text-center text-2xl font-semibold tracking-[0.4em] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              placeholder="••••••"
              aria-label={t('codeLabel')}
            />
          </div>
          <Button type="submit" className="w-full" disabled={checking || code.length < 6}>
            {checking ? t('checking') : t('verify')}
          </Button>
        </form>

        <p className="text-center text-sm text-muted-foreground">
          {t('noCodeYet')}{' '}
          <button
            type="button"
            onClick={onResend}
            disabled={resendState === 'sending'}
            className="font-medium text-foreground underline underline-offset-2 hover:text-primary disabled:opacity-50"
          >
            {resendState === 'sending' ? t('resending') : t('resend')}
          </button>
        </p>
      </div>
    </div>
  );
}
