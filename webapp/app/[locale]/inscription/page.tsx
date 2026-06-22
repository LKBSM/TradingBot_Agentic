import type { Metadata } from 'next';
import { RegisterForm } from '@/components/auth/RegisterForm';

export const metadata: Metadata = {
  title: 'Inscription',
  description: 'Créer un compte MIA Markets — accès anticipé, posture éducative.',
  robots: { index: false, follow: false },
};

export default function RegisterPage() {
  return (
    <div className="container-prose flex justify-center py-12 sm:py-16">
      <div className="w-full max-w-md space-y-6">
        <header className="space-y-1.5 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">Créer un compte</h1>
          <p className="text-sm text-muted-foreground">
            Accès anticipé réservé aux personnes majeures (18 ans ou plus).
          </p>
        </header>
        <RegisterForm />
      </div>
    </div>
  );
}
