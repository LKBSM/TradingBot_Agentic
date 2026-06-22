'use client';

import Link from 'next/link';
import * as React from 'react';
import { AuthError, requestPasswordReset } from '@/lib/auth/api-client';
import { Button } from '@/components/ui/button';
import { FormError, FormSuccess, TextField } from './fields';

/**
 * Password-reset request. The backend answers identically whether or not the
 * identifier matched (anti-enumeration), so this form always shows the same
 * neutral confirmation on success.
 */
export function ForgotPasswordForm() {
  const [error, setError] = React.useState<string | null>(null);
  const [done, setDone] = React.useState<string | null>(null);
  const [submitting, setSubmitting] = React.useState(false);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    const form = new FormData(e.currentTarget);
    setSubmitting(true);
    try {
      const res = await requestPasswordReset(
        String(form.get('identifier') ?? '').trim(),
      );
      setDone(res.message);
    } catch (err) {
      setError(err instanceof AuthError ? err.message : 'Demande impossible. Réessaie.');
    } finally {
      setSubmitting(false);
    }
  }

  if (done) {
    return (
      <div className="space-y-4">
        <FormSuccess message={done} />
        <Link
          href="/connexion"
          className="block text-center text-sm text-muted-foreground underline underline-offset-2 hover:text-foreground"
        >
          Retour à la connexion
        </Link>
      </div>
    );
  }

  return (
    <form onSubmit={onSubmit} className="space-y-4" noValidate>
      <FormError message={error} />
      <p className="text-sm text-muted-foreground">
        Indiquez votre nom d’utilisateur ou votre e-mail. Si un compte
        correspond, un lien de réinitialisation vous sera envoyé.
      </p>
      <TextField
        label="Nom d’utilisateur ou e-mail"
        name="identifier"
        autoComplete="username"
        required
      />
      <Button type="submit" className="w-full" disabled={submitting}>
        {submitting ? 'Envoi…' : 'Envoyer le lien'}
      </Button>
      <p className="text-center text-sm text-muted-foreground">
        <Link href="/connexion" className="underline underline-offset-2 hover:text-foreground">
          Retour à la connexion
        </Link>
      </p>
    </form>
  );
}
