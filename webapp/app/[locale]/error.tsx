'use client';

/**
 * Segment error boundary for every localized route (UI-02). Without it, a throw
 * in any client component (a disposed chart, malformed data slipping past a
 * guard) unmounts the whole tree and leaves a blank white screen with no way
 * out. Here we catch it, keep the chrome (this renders INSIDE the locale layout,
 * so Nav + Footer + theme stay), and offer a retry button (`reset()`) plus a
 * link home.
 *
 * I18N-1: fully translated (fr/en/es) — the whole screen must be in the active
 * language. The strings are an INLINE per-locale map read from the URL param,
 * NOT next-intl: an error boundary must never depend on the i18n provider, since
 * a fault in that very provider would double-fault the boundary. The disclaimer
 * mirrors the canonical `connexion.trust`, kept in sync here by hand.
 */

import * as React from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { Button } from '@/components/ui/button';

type Copy = {
  heading: string;
  body: string;
  reference: string;
  retry: string;
  back: string;
  trust: string;
};

const COPY: Record<'fr' | 'en' | 'es', Copy> = {
  fr: {
    heading: 'Une erreur est survenue',
    body: "Quelque chose s'est mal passé de notre côté. Tu peux réessayer — si le problème persiste, reviens à l'accueil.",
    reference: 'Référence',
    retry: 'Réessayer',
    back: "Retour à l'accueil",
    trust: 'Outil éducatif de lecture de marché. Ni signal de trading, ni conseil en investissement. 18+.',
  },
  en: {
    heading: 'Something went wrong',
    body: 'Something went wrong on our side. You can try again — if the problem persists, head back home.',
    reference: 'Reference',
    retry: 'Try again',
    back: 'Back to home',
    trust: 'Educational market-reading tool. Neither a trading signal nor investment advice. 18+.',
  },
  es: {
    heading: 'Se ha producido un error',
    body: 'Algo salió mal de nuestro lado. Puedes volver a intentarlo; si el problema persiste, vuelve al inicio.',
    reference: 'Referencia',
    retry: 'Reintentar',
    back: 'Volver al inicio',
    trust: 'Herramienta educativa de lectura de mercado. Ni señal de trading ni asesoramiento de inversión. 18+.',
  },
};

export default function LocaleError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const params = useParams();
  const raw = Array.isArray(params?.locale) ? params.locale[0] : params?.locale;
  const c = COPY[(raw as 'fr' | 'en' | 'es') in COPY ? (raw as 'fr' | 'en' | 'es') : 'fr'];

  React.useEffect(() => {
    // Surface the error in the console/observability; the digest ties a client
    // report back to the server log line for the same error.
    console.error('Route error boundary caught:', error);
  }, [error]);

  return (
    <div className="container-prose flex min-h-[60vh] flex-col items-center justify-center gap-6 py-16 text-center">
      <div className="space-y-3">
        <h1 className="text-2xl font-semibold text-foreground">{c.heading}</h1>
        <p className="text-muted-foreground">{c.body}</p>
        {error.digest ? (
          <p className="text-xs text-muted-foreground/70">
            {c.reference}&nbsp;: {error.digest}
          </p>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center justify-center gap-3">
        <Button onClick={() => reset()}>{c.retry}</Button>
        <Button variant="outline" asChild>
          <Link href="/">{c.back}</Link>
        </Button>
      </div>
      {/* Regulatory notice — one per page, never zero. Canonical disclaimer,
          inlined so the boundary never depends on the i18n provider. */}
      <p className="max-w-md text-xs text-muted-foreground/70">{c.trust}</p>
    </div>
  );
}
