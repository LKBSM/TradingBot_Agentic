'use client';

import { useRouter, useSearchParams } from 'next/navigation';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import {
  confirmEmailVerification,
  resendVerification,
} from '@/lib/auth/api-client';
import { useAuth } from '@/lib/auth/store';
import { useLocalizedHref } from '@/lib/i18n/href';
import { FormError, FormSuccess } from './fields';

/**
 * Email-verification screen (PAY-1/PAY-3), two modes on one page:
 *
 *  · WITH a ?token= (from the emailed link) → confirm it once, then forward to
 *    the plan-choice page (paying is the condition of entry).
 *  · WITHOUT a token (landed here right after signing up) → the "check your
 *    inbox" state with a RESEND button, so the failure mode "I never got the
 *    email" is never a dead end.
 */
export function EmailVerifier() {
  const t = useTranslations('auth.verifyEmail');
  const params = useSearchParams();
  const token = params.get('token') ?? '';
  const { account, refresh } = useAuth();
  const router = useRouter();
  const lh = useLocalizedHref();

  // Token-confirmation state (only meaningful when a token is present).
  const [state, setState] = React.useState<'verifying' | 'ok' | 'error'>(
    token ? 'verifying' : 'ok',
  );
  const ranRef = React.useRef(false);

  // Resend state (inbox mode).
  const [resendState, setResendState] = React.useState<
    'idle' | 'sending' | 'sent' | 'error'
  >('idle');

  React.useEffect(() => {
    if (!token) return; // inbox mode — nothing to confirm
    if (ranRef.current) return; // confirm exactly once (token is single-use)
    ranRef.current = true;
    confirmEmailVerification(token)
      .then(() => {
        setState('ok');
        void refresh();
      })
      .catch(() => setState('error'));
  }, [token, refresh]);

  // Once the email is CONFIRMED via the link, the mandatory next step is
  // choosing a plan — forward automatically. (Inbox mode, where state starts as
  // 'ok' with no token, must NOT auto-forward: the user still has to click the
  // link first.)
  React.useEffect(() => {
    if (!token || state !== 'ok') return;
    const id = setTimeout(() => router.push(lh('/abonnement')), 1200);
    return () => clearTimeout(id);
  }, [token, state, router, lh]);

  async function onResend() {
    setResendState('sending');
    try {
      await resendVerification();
      setResendState('sent');
    } catch {
      setResendState('error');
    }
  }

  // ── Token mode ──────────────────────────────────────────────────────────
  if (token) {
    return (
      <div className="pagewrap" style={{ maxWidth: 520 }}>
        <div className="card">
          <h1>{t('title')}</h1>
          {state === 'verifying' && (
            <p className="text-sm text-muted-foreground" aria-live="polite">
              {t('verifying')}
            </p>
          )}
          {state === 'ok' && (
            <div style={{ display: 'grid', gap: 12 }}>
              <FormSuccess message={t('success')} />
              <button
                type="button"
                className="btn primary"
                onClick={() => router.push(lh('/abonnement'))}
              >
                {t('cta')}
              </button>
            </div>
          )}
          {state === 'error' && <FormError message={t('error')} />}
        </div>
      </div>
    );
  }

  // ── Inbox mode (just signed up) ─────────────────────────────────────────
  return (
    <div className="pagewrap" style={{ maxWidth: 520 }}>
      <div className="card" style={{ display: 'grid', gap: 14 }}>
        <h1>{t('inboxTitle')}</h1>
        <p className="text-sm text-muted-foreground">
          {account?.email
            ? t('inboxBody', { email: account.email })
            : t('inboxBodyNoEmail')}
        </p>
        {resendState === 'sent' && <FormSuccess message={t('resent')} />}
        {resendState === 'error' && <FormError message={t('resendError')} />}
        <div style={{ display: 'grid', gap: 8 }}>
          <button
            type="button"
            className="btn"
            onClick={onResend}
            disabled={resendState === 'sending'}
          >
            {resendState === 'sending' ? t('resending') : t('resend')}
          </button>
          <button
            type="button"
            className="btn primary"
            onClick={() => router.push(lh('/abonnement'))}
          >
            {t('goToPlans')}
          </button>
        </div>
      </div>
    </div>
  );
}
