import type { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { ConditionsDocument } from '@/components/legal/ConditionsDocument';

export const metadata: Metadata = {
  title: 'Conditions d’utilisation',
  description:
    'Conditions Générales d’Utilisation de MIA Markets — service d’information, posture éducative, avertissement sur les risques.',
};

/**
 * /conditions — renders the canonical CGU document
 * (docs/legal/conditions-utilisation.md) TEL QUEL via the backend endpoint. The
 * text is never rewritten here; only the markdown is formatted for the web.
 */
export default function ConditionsPage() {
  return (
    <div className="container-prose py-12 sm:py-16">
      <Link
        href="/"
        className="mb-6 inline-flex items-center gap-1.5 text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
      >
        <ArrowLeft className="h-3.5 w-3.5" aria-hidden />
        Retour à l’accueil
      </Link>
      <ConditionsDocument />
    </div>
  );
}
