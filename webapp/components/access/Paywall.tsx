'use client';

import * as React from 'react';
import Link from 'next/link';
import { Lock } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

export interface PaywallProps {
  /** Short title — defaults to a generic "réservé aux abonnés". */
  title?: string;
  /** One-line explanation of what's behind the wall. */
  description?: string;
  /** When true, also offer a "se connecter" link (caller isn't authenticated). */
  needsLogin?: boolean;
  /** Locale-aware base prefix (e.g. "" for fr, "/en" for en). */
  basePrefix?: string;
  className?: string;
}

/**
 * Clean upsell surface shown in place of gated content. NEVER a raw error: it
 * explains the limit and invites the visitor to subscribe (or log in). The
 * server-side gate is what actually enforces access — this is the friendly face.
 */
export function Paywall({
  title = 'Réservé aux abonnés',
  description = 'Passe à l’abonnement pour débloquer cette fonctionnalité.',
  needsLogin = false,
  basePrefix = '',
  className,
}: PaywallProps) {
  return (
    <Card
      className={['mx-auto max-w-md p-8 text-center', className]
        .filter(Boolean)
        .join(' ')}
      role="region"
      aria-label="Contenu réservé aux abonnés"
    >
      <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
        <Lock className="h-5 w-5 text-primary" aria-hidden />
      </div>
      <h2 className="text-lg font-semibold text-foreground">{title}</h2>
      <p className="mt-2 text-sm text-muted-foreground">{description}</p>
      <div className="mt-6 flex flex-col items-center gap-2">
        <Button asChild className="w-full sm:w-auto">
          <Link href={`${basePrefix}/abonnement`}>Voir les abonnements</Link>
        </Button>
        {needsLogin ? (
          <Button asChild variant="ghost" size="sm" className="w-full sm:w-auto">
            <Link href={`${basePrefix}/connexion`}>J’ai déjà un compte — me connecter</Link>
          </Button>
        ) : null}
      </div>
    </Card>
  );
}
