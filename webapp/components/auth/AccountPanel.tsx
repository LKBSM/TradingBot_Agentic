'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import { ShieldCheck } from 'lucide-react';
import { AuthError, updateProfile } from '@/lib/auth/api-client';
import { useAuth } from '@/lib/auth/store';
import { useLocalizedHref } from '@/lib/i18n/href';
import { Button } from '@/components/ui/button';
import { FormError, FormSuccess, TextField } from './fields';

/**
 * Authenticated account panel: identity, role badge, recorded consents
 * (version + timestamp), email update, and logout. Redirects to /connexion when
 * the session probe resolves to "logged out".
 */
export function AccountPanel() {
  const t = useTranslations('auth');
  const { account, loading, probeFailed, logout, refresh } = useAuth();
  const router = useRouter();
  const lh = useLocalizedHref();
  const [error, setError] = React.useState<string | null>(null);
  const [success, setSuccess] = React.useState<string | null>(null);
  const [saving, setSaving] = React.useState(false);
  const savingRef = React.useRef(false);

  // Redirect to login ONLY on a confirmed logged-out state (probe returned 401).
  // A network/5xx failure (probeFailed) must NOT eject a possibly-valid user —
  // we show a retry instead (AUTH-03/AUTH-09).
  React.useEffect(() => {
    if (!loading && account === null && !probeFailed) router.replace(lh('/connexion'));
  }, [loading, account, probeFailed, router, lh]);

  if (!loading && account === null && probeFailed) {
    return (
      <div className="space-y-4">
        <FormError message={t('account.sessionError')} />
        <Button variant="outline" onClick={() => refresh()}>
          {t('account.retry')}
        </Button>
      </div>
    );
  }

  if (loading || account === null) {
    return <p className="text-sm text-muted-foreground">{t('account.loading')}</p>;
  }

  async function onSaveEmail(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    // Guard against a double submit (Enter + click) racing two requests before
    // React flips `saving` — the ref updates synchronously (AUTH-08).
    if (savingRef.current) return;
    savingRef.current = true;
    setError(null);
    setSuccess(null);
    const form = new FormData(e.currentTarget);
    setSaving(true);
    try {
      await updateProfile(String(form.get('email') ?? '').trim());
      // The server re-issued the session cookie on email change (AUTH-14), so a
      // follow-up refresh() reflects the new state without ever nulling the
      // account on a transient error (refresh is non-destructive now, AUTH-03).
      await refresh();
      setSuccess(t('account.emailUpdated'));
    } catch (err) {
      setError(err instanceof AuthError ? err.message : t('account.updateError'));
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  }

  async function onLogout() {
    await logout();
    router.push(lh('/'));
  }

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            {account.username}
          </h1>
          <p className="text-sm text-muted-foreground">{account.email}</p>
        </div>
        {account.role === 'owner' && (
          <span className="inline-flex items-center gap-1 rounded-full border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-xs font-medium text-amber-600">
            <ShieldCheck className="h-3.5 w-3.5" aria-hidden />
            {t('account.ownerBadge')}
          </span>
        )}
      </div>

      <section className="space-y-4 rounded-lg border border-border/60 p-5">
        <h2 className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
          {t('account.editEmailTitle')}
        </h2>
        <form onSubmit={onSaveEmail} className="space-y-3" noValidate>
          <FormError message={error} />
          <FormSuccess message={success} />
          <TextField
            label={t('account.emailLabel')}
            name="email"
            type="email"
            defaultValue={account.email}
            autoComplete="email"
            required
          />
          <Button type="submit" variant="secondary" disabled={saving}>
            {saving ? t('account.saving') : t('account.save')}
          </Button>
        </form>
      </section>

      <section className="space-y-3 rounded-lg border border-border/60 p-5">
        <h2 className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
          {t('account.consentsTitle')}
        </h2>
        <ul className="space-y-2 text-sm">
          {account.consents.length === 0 && (
            <li className="text-muted-foreground">{t('account.noConsents')}</li>
          )}
          {account.consents.map((c) => (
            <li key={`${c.doc}-${c.version}`} className="flex items-center justify-between gap-3">
              <span className="capitalize text-foreground">
                {c.doc === 'terms' ? t('account.docTerms') : t('account.docPrivacy')}
              </span>
              <span className="text-xs text-muted-foreground">
                {t('account.consentMeta', {
                  version: c.version,
                  date: c.accepted_at.replace('T', ' '),
                })}
              </span>
            </li>
          ))}
        </ul>
        <p className="text-xs text-muted-foreground">
          {t.rich('account.seeDocs', {
            terms: (chunks) => (
              <Link href={lh('/conditions')} className="underline underline-offset-2 hover:text-foreground">
                {chunks}
              </Link>
            ),
            privacy: (chunks) => (
              <Link href={lh('/confidentialite')} className="underline underline-offset-2 hover:text-foreground">
                {chunks}
              </Link>
            ),
          })}
        </p>
      </section>

      <div>
        <Button variant="outline" onClick={onLogout}>
          {t('account.logout')}
        </Button>
      </div>
    </div>
  );
}
