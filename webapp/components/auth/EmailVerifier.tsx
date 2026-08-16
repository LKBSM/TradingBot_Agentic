'use client';

import { useRouter, useSearchParams } from 'next/navigation';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import { confirmEmailVerification } from '@/lib/auth/api-client';
import { useAuth } from '@/lib/auth/store';
import { useLocalizedHref } from '@/lib/i18n/href';
import { FormError, FormSuccess } from './fields';

/**
 * Confirms the email-verification token from the emailed link (PAY-1). Runs the
 * confirmation once on mount, then shows success (with a CTA to the app) or a
 * clear error. Refreshes the auth state so the verified flag is reflected.
 */
export function EmailVerifier() {
  const t = useTranslations('auth.verifyEmail');
  const params = useSearchParams();
  const token = params.get('token') ?? '';
  const { refresh } = useAuth();
  const router = useRouter();
  const lh = useLocalizedHref();
  const [state, setState] = React.useState<'verifying' | 'ok' | 'error'>('verifying');
  const ranRef = React.useRef(false);

  React.useEffect(() => {
    if (ranRef.current) return; // confirm exactly once (token is single-use)
    ranRef.current = true;
    if (!token) {
      setState('error');
      return;
    }
    confirmEmailVerification(token)
      .then(() => {
        setState('ok');
        void refresh();
      })
      .catch(() => setState('error'));
  }, [token, refresh]);

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
              onClick={() => router.push(lh('/app'))}
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
