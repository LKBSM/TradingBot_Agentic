'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import * as React from 'react';
import { AuthError } from '@/lib/auth/api-client';
import { useAuth } from '@/lib/auth/store';
import { Button } from '@/components/ui/button';
import { CheckField, FormError, TextField } from './fields';

/**
 * Registration form. Enforces, client-side, the same gates the backend enforces
 * server-side (defence in depth, never the only check): 18+ self-declaration +
 * explicit acceptance of the Conditions and the Privacy Policy. The version and
 * timestamp of those consents are recorded server-side at account creation.
 */
export function RegisterForm() {
  const { register } = useAuth();
  const router = useRouter();
  const [error, setError] = React.useState<string | null>(null);
  const [submitting, setSubmitting] = React.useState(false);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    const form = new FormData(e.currentTarget);
    const ageConfirmed = form.get('age_confirmed') === 'on';
    const acceptTerms = form.get('accept_terms') === 'on';
    const acceptPrivacy = form.get('accept_privacy') === 'on';

    if (!ageConfirmed) {
      setError('Vous devez déclarer avoir 18 ans ou plus.');
      return;
    }
    if (!acceptTerms || !acceptPrivacy) {
      setError(
        'Vous devez accepter les Conditions d’utilisation et la Politique de confidentialité.',
      );
      return;
    }

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
      router.push('/compte');
    } catch (err) {
      setError(
        err instanceof AuthError ? err.message : 'Inscription impossible. Réessaie.',
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="space-y-4" noValidate>
      <FormError message={error} />
      <TextField
        label="Nom d’utilisateur"
        name="username"
        autoComplete="username"
        required
        minLength={3}
        maxLength={32}
        hint="3 à 32 caractères."
      />
      <TextField
        label="Adresse e-mail"
        name="email"
        type="email"
        autoComplete="email"
        required
      />
      <TextField
        label="Mot de passe"
        name="password"
        type="password"
        autoComplete="new-password"
        required
        minLength={10}
        hint="Au moins 10 caractères."
      />

      <fieldset className="space-y-2.5 rounded-md border border-border/60 bg-muted/20 p-3">
        <legend className="px-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Déclarations obligatoires
        </legend>
        <CheckField
          name="age_confirmed"
          label="Je déclare avoir 18 ans ou plus."
        />
        <CheckField
          name="accept_terms"
          label={
            <>
              J’accepte les{' '}
              <Link href="/conditions" className="underline underline-offset-2 hover:text-foreground">
                Conditions d’utilisation
              </Link>
              .
            </>
          }
        />
        <CheckField
          name="accept_privacy"
          label={
            <>
              J’accepte la{' '}
              <Link href="/confidentialite" className="underline underline-offset-2 hover:text-foreground">
                Politique de confidentialité
              </Link>
              .
            </>
          }
        />
      </fieldset>

      <Button type="submit" className="w-full" disabled={submitting}>
        {submitting ? 'Création…' : 'Créer mon compte'}
      </Button>

      <p className="text-center text-sm text-muted-foreground">
        Déjà un compte ?{' '}
        <Link href="/connexion" className="underline underline-offset-2 hover:text-foreground">
          Se connecter
        </Link>
      </p>
    </form>
  );
}
