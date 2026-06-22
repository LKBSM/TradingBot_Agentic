'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import * as React from 'react';
import { ShieldCheck } from 'lucide-react';
import { AuthError, updateProfile } from '@/lib/auth/api-client';
import { useAuth } from '@/lib/auth/store';
import { Button } from '@/components/ui/button';
import { FormError, FormSuccess, TextField } from './fields';

/**
 * Authenticated account panel: identity, role badge, recorded consents
 * (version + timestamp), email update, and logout. Redirects to /connexion when
 * the session probe resolves to "logged out".
 */
export function AccountPanel() {
  const { account, loading, logout, refresh } = useAuth();
  const router = useRouter();
  const [error, setError] = React.useState<string | null>(null);
  const [success, setSuccess] = React.useState<string | null>(null);
  const [saving, setSaving] = React.useState(false);

  React.useEffect(() => {
    if (!loading && account === null) router.replace('/connexion');
  }, [loading, account, router]);

  if (loading || account === null) {
    return <p className="text-sm text-muted-foreground">Chargement…</p>;
  }

  async function onSaveEmail(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    const form = new FormData(e.currentTarget);
    setSaving(true);
    try {
      await updateProfile(String(form.get('email') ?? '').trim());
      await refresh();
      setSuccess('Adresse e-mail mise à jour.');
    } catch (err) {
      setError(err instanceof AuthError ? err.message : 'Mise à jour impossible.');
    } finally {
      setSaving(false);
    }
  }

  async function onLogout() {
    await logout();
    router.push('/');
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
            Propriétaire
          </span>
        )}
      </div>

      <section className="space-y-4 rounded-lg border border-border/60 p-5">
        <h2 className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
          Modifier mon e-mail
        </h2>
        <form onSubmit={onSaveEmail} className="space-y-3" noValidate>
          <FormError message={error} />
          <FormSuccess message={success} />
          <TextField
            label="Adresse e-mail"
            name="email"
            type="email"
            defaultValue={account.email}
            autoComplete="email"
            required
          />
          <Button type="submit" variant="secondary" disabled={saving}>
            {saving ? 'Enregistrement…' : 'Enregistrer'}
          </Button>
        </form>
      </section>

      <section className="space-y-3 rounded-lg border border-border/60 p-5">
        <h2 className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
          Consentements enregistrés
        </h2>
        <ul className="space-y-2 text-sm">
          {account.consents.length === 0 && (
            <li className="text-muted-foreground">Aucun consentement enregistré.</li>
          )}
          {account.consents.map((c) => (
            <li key={`${c.doc}-${c.version}`} className="flex items-center justify-between gap-3">
              <span className="capitalize text-foreground">
                {c.doc === 'terms' ? 'Conditions d’utilisation' : 'Confidentialité'}
              </span>
              <span className="text-xs text-muted-foreground">
                v{c.version} · {c.accepted_at.replace('T', ' ')}
              </span>
            </li>
          ))}
        </ul>
        <p className="text-xs text-muted-foreground">
          Voir les{' '}
          <Link href="/conditions" className="underline underline-offset-2 hover:text-foreground">
            Conditions
          </Link>{' '}
          et la{' '}
          <Link href="/confidentialite" className="underline underline-offset-2 hover:text-foreground">
            Politique de confidentialité
          </Link>
          .
        </p>
      </section>

      <div>
        <Button variant="outline" onClick={onLogout}>
          Se déconnecter
        </Button>
      </div>
    </div>
  );
}
