'use client';

/**
 * Segment error boundary for every localized route (UI-02). Without it, a throw
 * in any client component (a disposed chart, malformed data slipping past a
 * guard) unmounts the whole tree and leaves a blank white screen with no way
 * out. Here we catch it, keep the chrome (this renders INSIDE the locale layout,
 * so Nav + Footer + theme stay), and offer a Réessayer button (`reset()`) plus a
 * link home. FR-only copy — V1 ships FR.
 */

import * as React from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';

export default function LocaleError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  React.useEffect(() => {
    // Surface the error in the console/observability; the digest ties a client
    // report back to the server log line for the same error.
    console.error('Route error boundary caught:', error);
  }, [error]);

  return (
    <div className="container-prose flex min-h-[60vh] flex-col items-center justify-center gap-6 py-16 text-center">
      <div className="space-y-3">
        <h1 className="text-2xl font-semibold text-foreground">
          Une erreur est survenue
        </h1>
        <p className="text-muted-foreground">
          Quelque chose s&apos;est mal passé de notre côté. Tu peux réessayer —
          si le problème persiste, reviens à l&apos;accueil.
        </p>
        {error.digest ? (
          <p className="text-xs text-muted-foreground/70">
            Référence&nbsp;: {error.digest}
          </p>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center justify-center gap-3">
        <Button onClick={() => reset()}>Réessayer</Button>
        <Button variant="outline" asChild>
          <Link href="/">Retour à l&apos;accueil</Link>
        </Button>
      </div>
    </div>
  );
}
