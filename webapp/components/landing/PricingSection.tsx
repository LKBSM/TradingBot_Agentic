'use client';

import { useState } from 'react';
import { Check, ExternalLink } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';

/**
 * Section L5.1 — Pricing.
 *
 * Décision fondateur 2026-07-07 : plan unique payant, tout l'outil inclus,
 * avec bascule mensuel / annuel. Les anciens tiers FREE / 9 € / 19 € sont
 * retirés — les démos de la landing restent l'unique surface gratuite.
 *
 *   Mensuel : 49,99 $ / mois, sans engagement.
 *   Annuel  : 39,99 $ / mois (facturé 479,88 $ / an), soit −20 %.
 *
 * Règle éditoriale (nettoyage claims 2026-07-04, garde-fou
 * `tests/claims-cleanup.test.ts`) : aucun compteur de places, aucune
 * référence réglementaire non vérifiée, aucune promesse (remboursement,
 * périmètre géographique) non implémentée. On ne promet donc PAS de
 * remboursement tant qu'il n'est pas câblé côté produit.
 */
const MONTHLY_PRICE = '49,99';
const ANNUAL_PRICE = '39,99';
const ANNUAL_TOTAL = '479,88';

const FEATURES: ReadonlyArray<string> = [
  'Lectures illimitées XAU/USD + EUR/USD · M15 → H4',
  'Toutes les sections : structure SMC, régime, volatilité, événements',
  'Chatbot M.I.A Agent illimité (Claude, contexte de la lecture injecté)',
  'Scanner multi-combos + page Zones (cycle de vie OB / FVG)',
  'Bannière événement macro ≤ 4 h',
  'Synthèse experte détaillée + sources RAG académiques',
  'Historique complet de vos lectures',
  'Annulation en un clic',
];

type Cadence = 'monthly' | 'annual';

export function PricingSection() {
  const [cadence, setCadence] = useState<Cadence>('annual');
  const isAnnual = cadence === 'annual';
  const price = isAnnual ? ANNUAL_PRICE : MONTHLY_PRICE;

  return (
    <section
      id="tarifs"
      aria-labelledby="pricing-title"
      className="container-wide space-y-10 py-16 sm:py-20"
    >
      <header className="max-w-2xl space-y-3">
        <Badge
          variant="outline"
          className="border-sentinel-bull/40 text-[11px] uppercase tracking-wider text-sentinel-bull"
        >
          Accès anticipé
        </Badge>
        <h2
          id="pricing-title"
          className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl"
        >
          Un seul plan. Tout l&apos;outil, sans demi-mesure.
        </h2>
        <p className="max-w-2xl text-pretty text-muted-foreground">
          Pas d&apos;essai déguisé, pas de tier qui ampute l&apos;outil : un
          accès complet, résiliable en un clic. Les démos de cette page sont
          gratuites — vous voyez exactement ce que vous obtenez. Voir notre
          posture dans la section{' '}
          <a href="#honnetete" className="underline-offset-2 hover:underline">
            Honnêteté
          </a>
          .
        </p>
      </header>

      {/* Bascule mensuel / annuel */}
      <div
        className="inline-flex items-center rounded-full border border-border/70 bg-muted/40 p-1 text-sm"
        role="group"
        aria-label="Choisir la fréquence de facturation"
      >
        <button
          type="button"
          onClick={() => setCadence('monthly')}
          aria-pressed={!isAnnual}
          className={cn(
            'rounded-full px-4 py-1.5 font-medium transition-colors',
            !isAnnual
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground',
          )}
        >
          Mensuel
        </button>
        <button
          type="button"
          onClick={() => setCadence('annual')}
          aria-pressed={isAnnual}
          className={cn(
            'inline-flex items-center gap-1.5 rounded-full px-4 py-1.5 font-medium transition-colors',
            isAnnual
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground',
          )}
        >
          Annuel
          <span className="rounded-full bg-sentinel-bull/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-sentinel-bull">
            −20 %
          </span>
        </button>
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)] lg:items-center">
        {/* Carte plan unique */}
        <Card className="border-primary/50 shadow-md">
          <CardContent className="flex flex-col gap-5 p-6 sm:p-7">
            <header className="space-y-2">
              <div className="flex items-center gap-2">
                <h3 className="text-base font-semibold tracking-tight">
                  Accès intégral MIA
                </h3>
                <Badge variant="default" className="text-[10px]">
                  Tout inclus
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                Un seul abonnement, l&apos;outil complet — lecture, chatbot,
                scanner, zones.
              </p>
            </header>

            <div className="space-y-1">
              <p className="flex items-baseline gap-1 tabular-nums">
                <span className="text-4xl font-semibold">{price}&nbsp;$</span>
                <span className="text-sm text-muted-foreground">/ mois</span>
              </p>
              <p className="text-xs text-muted-foreground">
                {isAnnual
                  ? `Facturé ${ANNUAL_TOTAL} $ par an — soit 120 $ d'économie (−20 %).`
                  : 'Sans engagement, résiliable à tout moment.'}
              </p>
            </div>

            <ul className="space-y-2 text-sm">
              {FEATURES.map((feature) => (
                <li key={feature} className="flex items-start gap-2">
                  <Check
                    className="mt-0.5 h-4 w-4 shrink-0 text-sentinel-bull"
                    aria-hidden
                  />
                  <span>{feature}</span>
                </li>
              ))}
            </ul>

            <Button
              type="button"
              variant="default"
              className="w-full"
              disabled
              aria-disabled="true"
              title="Inscription disponible après l'intégration backend"
            >
              S&apos;abonner
            </Button>
          </CardContent>
        </Card>

        {/* Réassurance à droite — uniquement des faits vérifiables. */}
        <ul className="space-y-4 text-sm text-muted-foreground lg:pl-4">
          <li className="flex items-start gap-3">
            <Check className="mt-0.5 h-4 w-4 shrink-0 text-sentinel-bull" aria-hidden />
            <span>
              <strong className="text-foreground">Aucune carte piégée.</strong>{' '}
              Annulation en un clic depuis votre compte, sans relance ni
              questionnaire de rétention.
            </span>
          </li>
          <li className="flex items-start gap-3">
            <Check className="mt-0.5 h-4 w-4 shrink-0 text-sentinel-bull" aria-hidden />
            <span>
              <strong className="text-foreground">Sans engagement de durée.</strong>{' '}
              L&apos;offre mensuelle se résilie à tout moment ; l&apos;annuelle
              reste active jusqu&apos;à la fin de la période payée.
            </span>
          </li>
          <li className="flex items-start gap-3">
            <Check className="mt-0.5 h-4 w-4 shrink-0 text-sentinel-bull" aria-hidden />
            <span>
              <strong className="text-foreground">Vos données restent vôtres.</strong>{' '}
              Lectures et conversations privées, jamais revendues, jamais
              croisées à des fins publicitaires.
            </span>
          </li>
          <li className="flex items-start gap-3">
            <Check className="mt-0.5 h-4 w-4 shrink-0 text-sentinel-bull" aria-hidden />
            <span>
              <strong className="text-foreground">Testez avant de payer.</strong>{' '}
              Les démos de cette page sont interactives et gratuites — vous
              voyez exactement ce que vous achetez.
            </span>
          </li>
        </ul>
      </div>

      {/* Bloc B2B / Institutional discret — contact, pas de prix affiché. */}
      <aside className="rounded-2xl border border-dashed border-border/70 bg-muted/30 p-5 sm:p-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="max-w-xl space-y-1">
            <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Pour les équipes pro · API B2B
            </p>
            <p className="text-sm text-foreground">
              Brokers, fonds, médias : intégration API REST + webhooks
              signés, SLA et onboarding accompagné — tarif sur demande.
            </p>
          </div>
          <Button asChild variant="outline" size="sm" className="shrink-0">
            <a href="mailto:contact@mia.markets?subject=D%C3%A9mo%20B2B">
              Réserver une démo
              <ExternalLink className="ml-1.5 h-3.5 w-3.5" aria-hidden />
            </a>
          </Button>
        </div>
      </aside>

      <p className="text-xs italic text-muted-foreground">
        Démonstration en accès anticipé. MIA Markets produit des analyses
        éditoriales contextuelles et non des recommandations personnalisées.
      </p>
    </section>
  );
}
