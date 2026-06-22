import type { Metadata } from 'next';
import { LoginForm } from '@/components/auth/LoginForm';

export const metadata: Metadata = {
  title: 'Connexion',
  description: 'Se connecter à votre compte MIA Markets.',
  robots: { index: false, follow: false },
};

export default function LoginPage() {
  return (
    <div className="container-prose flex justify-center py-12 sm:py-16">
      <div className="w-full max-w-md space-y-6">
        <header className="space-y-1.5 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">Connexion</h1>
          <p className="text-sm text-muted-foreground">
            Connectez-vous avec votre nom d’utilisateur ou votre e-mail.
          </p>
        </header>
        <LoginForm />
      </div>
    </div>
  );
}
