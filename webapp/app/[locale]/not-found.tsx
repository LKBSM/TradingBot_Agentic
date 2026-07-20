/**
 * Localized 404 (NAV-09). Renders inside the locale layout (Nav + Footer +
 * theme + fonts), so an unknown URL no longer drops to Next's bare, unstyled
 * default page. Also catches `notFound()` from the layout (unsupported locale).
 * FR-only copy — V1 ships FR.
 */

import Link from 'next/link';
import { Button } from '@/components/ui/button';

export default function NotFound() {
  return (
    <div className="container-prose flex min-h-[60vh] flex-col items-center justify-center gap-6 py-16 text-center">
      <div className="space-y-3">
        <p className="text-sm font-medium text-muted-foreground">Erreur 404</p>
        <h1 className="text-2xl font-semibold text-foreground">
          Cette page est introuvable
        </h1>
        <p className="text-muted-foreground">
          Le lien est peut-être périmé ou l&apos;adresse mal saisie.
        </p>
      </div>
      <Button asChild>
        <Link href="/">Retour à l&apos;accueil</Link>
      </Button>
    </div>
  );
}
