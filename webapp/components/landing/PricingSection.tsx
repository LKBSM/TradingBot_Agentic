import { Check, ExternalLink } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface Tier {
  id: string;
  name: string;
  price: string;
  cadence: string;
  pitch: string;
  features: ReadonlyArray<string>;
  cta: string;
  highlight?: boolean;
}

/**
 * Section L5.1 — Pricing post-pivot 2026-05-27.
 *
 * Décision officielle : FREE / 9 € / 19 € (cf. `pivot_positioning_2026_05_27`
 * + `decisions/2026-05-27_pivot_positioning_audit.md`). INSTITUTIONAL retiré
 * de la grille publique, remplacé par un bloc "Réserver une démo" Calendly
 * en bas de section.
 *
 * Cap volontaire 50 abonnés (bootstrap legal) → mention "Early Access".
 * Refund 30 j + annulation 1 clic = défense MiFID II / UE 2024/2811.
 */
const TIERS: ReadonlyArray<Tier> = [
  {
    id: 'free',
    name: 'Découverte',
    price: 'Gratuit',
    cadence: '',
    pitch: 'Goûtez la lecture, sans CB, sans engagement.',
    features: [
      '1 lecture XAU M15 par jour',
      'Verdict + jauge de conviction',
      '3 questions chatbot par jour',
      'Posture éducative, sans signaux',
    ],
    cta: 'Commencer',
  },
  {
    id: 'approfondie',
    name: 'Approfondie',
    price: '9 €',
    cadence: '/ mois',
    pitch: 'Une lecture complète, chaque jour.',
    features: [
      'XAU + EUR · M15 / H1',
      'Sections collapsibles complètes',
      'Chatbot illimité',
      'Bannière event ≤ 4 h',
      'Annulation en un clic',
    ],
    cta: "S'abonner",
    highlight: true,
  },
  {
    id: 'integrale',
    name: 'Intégrale',
    price: '19 €',
    cadence: '/ mois',
    pitch: 'Tout l’outil pour les power-users SMC + macro.',
    features: [
      'Tout Approfondie',
      'Décomposition 8 composantes (waterfall)',
      'Visualisation conformelle + sources RAG',
      'Historique 50 lectures + PnL paper individuel',
      'Accès anticipé futures fonctionnalités',
    ],
    cta: "S'abonner",
  },
];

export function PricingSection() {
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
          Early Access · 50 places
        </Badge>
        <h2
          id="pricing-title"
          className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl"
        >
          Trois entrées. Annulation en un clic. Remboursement 30 jours.
        </h2>
        <p className="max-w-2xl text-pretty text-muted-foreground">
          Pas d&apos;essai gratuit déguisé, pas de carte demandée sur le
          tier Découverte. Les tarifs Early Access restent valables tant que
          MIA n&apos;a pas atteint l&apos;edge mesurable (cf. section{' '}
          <a href="#honnetete" className="underline-offset-2 hover:underline">
            Honnêteté conformelle
          </a>).
        </p>
      </header>

      <div className="grid gap-5 sm:gap-6 lg:grid-cols-3">
        {TIERS.map((tier) => (
          <TierCard key={tier.id} tier={tier} />
        ))}
      </div>

      {/* Bloc B2B / Institutional discret — pas dans la grille pour éviter le
          decoy et l'effet "trop cher" sur les vrais tiers retail. */}
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
          <Button
            asChild
            variant="outline"
            size="sm"
            className="shrink-0"
          >
            <a
              href="https://calendly.com/mia-markets/demo"
              target="_blank"
              rel="noopener noreferrer"
            >
              Réserver une démo
              <ExternalLink className="ml-1.5 h-3.5 w-3.5" aria-hidden />
            </a>
          </Button>
        </div>
      </aside>

      <p className="text-xs italic text-muted-foreground">
        Démonstration paper-trading. MIA Markets produit des analyses
        éditoriales contextuelles et non des recommandations
        personnalisées (UE 2024/2811). Disponibilité géographique
        restreinte — voir bas de page.
      </p>
    </section>
  );
}

function TierCard({ tier }: { tier: Tier }) {
  return (
    <Card
      className={cn(
        'flex flex-col border-border/60 shadow-sm',
        tier.highlight && 'border-primary/50 shadow-md',
      )}
    >
      <CardContent className="flex h-full flex-col gap-5 p-5 sm:p-6">
        <header className="space-y-2">
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold tracking-tight">
              {tier.name}
            </h3>
            {tier.highlight && (
              <Badge variant="default" className="text-[10px]">
                Recommandé
              </Badge>
            )}
          </div>
          <p className="text-xs text-muted-foreground">{tier.pitch}</p>
        </header>
        <p className="flex items-baseline gap-1 tabular-nums">
          <span className="text-3xl font-semibold">{tier.price}</span>
          {tier.cadence && (
            <span className="text-sm text-muted-foreground">
              {tier.cadence}
            </span>
          )}
        </p>
        <ul className="flex-1 space-y-2 text-sm">
          {tier.features.map((feature) => (
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
          variant={tier.highlight ? 'default' : 'outline'}
          className="w-full"
          disabled
          aria-disabled="true"
          title="Inscription disponible après l'intégration backend"
        >
          {tier.cta}
        </Button>
      </CardContent>
    </Card>
  );
}
