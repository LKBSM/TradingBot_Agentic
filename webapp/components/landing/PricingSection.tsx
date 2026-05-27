import { Check } from 'lucide-react';
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
  decoy?: boolean;
}

// LEGAL-PENDING: pricing labels, feature wording and CTAs are placeholders
// pending the legal terminal's review on "analyses" vs "signaux" wording,
// MiFID disclosure_mode=qualitative, refund policy phrasing, and dual-trial
// presentation (DG-070 + DG-073 + DG-079 + DG-083 + DG-084).
const TIERS: ReadonlyArray<Tier> = [
  {
    id: 'free',
    name: 'Découverte',
    price: 'Gratuit',
    cadence: '',
    pitch: "Goûtez la lecture, sans CB, sans engagement.",
    features: [
      '1 lecture XAU M15 par jour',
      'Verdict + jauge de conviction',
      '3 questions chatbot par jour',
      'Posture éducative, sans signaux',
    ],
    cta: 'Commencer',
  },
  {
    id: 'analyst',
    name: 'Analyste',
    price: '29 €',
    cadence: '/ mois',
    pitch: 'L’offre quotidienne du retail actif.',
    features: [
      'XAU + EUR · M15 / H1 / H4',
      'Sections collapsibles complètes',
      'Chatbot illimité',
      'Track-record public mensuel',
      'Essai 14 jours sans CB',
    ],
    cta: "S'abonner",
    highlight: true,
  },
  {
    id: 'strategist',
    name: 'Stratège',
    price: '79 €',
    cadence: '/ mois',
    pitch: 'Pour les power-users SMC et macros.',
    features: [
      'Tout Analyste',
      'Décomposition 8 composantes (waterfall)',
      'Visualisation conformelle + sources RAG',
      'Bannière event ≤ 4 h prioritaire',
      'Historique 50 lectures + PnL paper individuel',
    ],
    cta: "S'abonner",
  },
  {
    id: 'institutional',
    name: 'Institutionnel',
    price: '1 990 €',
    cadence: '/ mois',
    pitch: 'API B2B brokers + SLA + onboarding.',
    features: [
      'API REST + webhooks signés',
      'SLA 99.5 % uptime',
      'Onboarding accompagné',
      'Volume illimité, multi-instruments',
      'Démo personnalisée Calendly',
    ],
    cta: 'Demander une démo',
    decoy: true,
  },
];

export function PricingSection() {
  return (
    <section
      id="tarifs"
      aria-labelledby="pricing-title"
      className="container-prose space-y-8 py-12 sm:py-16"
    >
      <header className="space-y-2">
        <h2
          id="pricing-title"
          className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl"
        >
          Trois formules retail, une API institutionnelle.
        </h2>
        <p className="max-w-2xl text-muted-foreground">
          Essai 14 jours sans carte bancaire sur Analyste et Stratège.
          Annulation en un clic, remboursement intégral sous 30 jours.
        </p>
      </header>

      <div className="grid gap-4 lg:grid-cols-4">
        {TIERS.map((tier) => (
          <TierCard key={tier.id} tier={tier} />
        ))}
      </div>

      {/* LEGAL-PENDING: trial / refund / mediation wording — to be replaced
          by the legal terminal output (L.612-1 médiateur CM2C, Hamon 14j,
          MiFID disclosure). */}
      <p className="text-xs italic text-muted-foreground">
        Démonstration paper-trading. MIA Markets produit des analyses
        éditoriales contextuelles et non des recommandations personnalisées.
        Disponibilité géographique restreinte (US, Québec, Royaume-Uni et
        juridictions OFAC exclues).
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
            {tier.decoy && (
              <Badge variant="secondary" className="text-[10px]">
                B2B
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
          title="Inscription disponible après l'intégration backend (V2)"
        >
          {tier.cta}
        </Button>
      </CardContent>
    </Card>
  );
}
