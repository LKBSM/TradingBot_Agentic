import type { Metadata } from 'next';
import { AccountPanel } from '@/components/auth/AccountPanel';

export const metadata: Metadata = {
  title: 'Mon compte',
  description: 'Gérer votre compte MIA Markets.',
  robots: { index: false, follow: false },
};

export default function AccountPage() {
  return (
    <div className="container-prose py-12 sm:py-16">
      <AccountPanel />
    </div>
  );
}
