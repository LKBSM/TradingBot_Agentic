'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import * as React from 'react';
import { AuthError } from '@/lib/auth/api-client';
import { useAuth } from '@/lib/auth/store';
import { Button } from '@/components/ui/button';
import { FormError, TextField } from './fields';

/** Login form — identifier is a username OR an email (single field). */
export function LoginForm() {
  const { login } = useAuth();
  const router = useRouter();
  const [error, setError] = React.useState<string | null>(null);
  const [submitting, setSubmitting] = React.useState(false);

  // After login, go straight to the product. Honor a ?next= return path (set by
  // the login-wall redirect) when it's a safe internal path, else land on /app.
  // Read from window at submit time (client-only) to avoid the useSearchParams
  // Suspense-boundary requirement on the /connexion page build.
  function resolveDestination(): string {
    if (typeof window === 'undefined') return '/app';
    const next = new URLSearchParams(window.location.search).get('next');
    return next && next.startsWith('/') ? next : '/app';
  }

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    const form = new FormData(e.currentTarget);
    setSubmitting(true);
    try {
      await login({
        identifier: String(form.get('identifier') ?? '').trim(),
        password: String(form.get('password') ?? ''),
      });
      router.push(resolveDestination());
    } catch (err) {
      setError(
        err instanceof AuthError ? err.message : 'Connexion impossible. Réessaie.',
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="space-y-4" noValidate>
      <FormError message={error} />
      <TextField
        label="Nom d’utilisateur ou e-mail"
        name="identifier"
        autoComplete="username"
        required
      />
      <TextField
        label="Mot de passe"
        name="password"
        type="password"
        autoComplete="current-password"
        required
      />
      <div className="text-right">
        <Link
          href="/mot-de-passe-oublie"
          className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
        >
          Mot de passe oublié ?
        </Link>
      </div>
      <Button type="submit" className="w-full" disabled={submitting}>
        {submitting ? 'Connexion…' : 'Se connecter'}
      </Button>
      <p className="text-center text-sm text-muted-foreground">
        Pas encore de compte ?{' '}
        <Link href="/inscription" className="underline underline-offset-2 hover:text-foreground">
          Créer un compte
        </Link>
      </p>
    </form>
  );
}
