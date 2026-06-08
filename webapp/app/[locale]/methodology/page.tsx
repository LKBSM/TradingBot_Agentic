import type { Metadata } from 'next';
import Link from 'next/link';
import { ShieldCheck, ArrowLeft } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { MethodologySection } from '@/components/methodology/MethodologySection';
import { ConceptCard } from '@/components/methodology/ConceptCard';
import { ScoreFormula } from '@/components/methodology/ScoreFormula';
import {
  DATA_SOURCE,
  ENGAGEMENT_QUOTE,
  NEVER_DO,
  SCORE_FORMULAS,
  SMC_CONCEPTS,
} from '@/lib/methodology/content';

export const metadata: Metadata = {
  title: 'Méthodologie — Comment MIA Markets décrit le marché',
  description:
    'Documentation technique transparente : comment notre indicateur détecte les structures SMC (Order Block, Fair Value Gap, cassures), comment nous décrivons les éléments affichés, notre source de données, et ce que nous ne faisons pas.',
};

const TOC = [
  { href: '#engagement', label: 'Notre engagement' },
  { href: '#concepts', label: 'Concepts de structure' },
  { href: '#scores', label: 'Ce que nous décrivons' },
  { href: '#donnees', label: 'Source de données' },
  { href: '#limites', label: 'Ce que nous ne faisons pas' },
  { href: '#attributions', label: 'Attributions' },
] as const;

/**
 * Page /methodology (Chantier 5.D) — documentation algorithmique transparente.
 *
 * Descriptive et technique, JAMAIS promotionnelle (niveau 1.5 strict). Elle
 * répond à la curiosité légitime « comment l'algorithme calcule ça ? » sans
 * jamais prétendre prédire. Contenu data-driven (lib/methodology/content.ts),
 * termes partagés avec les tooltips ⓘ via le glossaire central.
 */
export default function MethodologyPage() {
  return (
    <div className="container-prose py-12 sm:py-16">
      <header className="space-y-4">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
        >
          <ArrowLeft className="h-3.5 w-3.5" aria-hidden />
          Retour à l’accueil
        </Link>
        <Badge
          variant="outline"
          className="text-[11px] uppercase tracking-wider"
        >
          <ShieldCheck className="mr-1 h-3 w-3" aria-hidden />
          Méthodologie
        </Badge>
        <h1 className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl lg:text-4xl">
          Comment MIA Markets décrit le marché
        </h1>
        <p className="max-w-2xl text-pretty text-muted-foreground">
          Cette page explique, sans jargon, comment notre indicateur repère les
          structures de marché et ce que signifie chaque élément affiché. C’est
          une documentation descriptive : on explique ce que le moteur
          <em> observe</em>, jamais ce qu’il <em>prédirait</em>.
        </p>
        <nav aria-label="Sommaire" className="flex flex-wrap gap-2 pt-2">
          {TOC.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="rounded-full border border-border/60 px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-foreground/40 hover:text-foreground"
            >
              {item.label}
            </a>
          ))}
        </nav>
      </header>

      <MethodologySection
        id="engagement"
        title="Notre engagement"
        intro="MIA Markets est un outil de compréhension du marché, pas un système de trading. Notre posture est descriptive (niveau « lecture augmentée ») : nous décrivons une structure, nous ne recommandons jamais une action."
      >
        <figure>
          <blockquote className="border-l-2 border-sentinel-warn pl-5 text-balance text-base italic text-foreground">
            {ENGAGEMENT_QUOTE}
          </blockquote>
          <figcaption className="mt-3 text-xs text-muted-foreground">
            — Engagement public MIA Markets, 27 mai 2026
          </figcaption>
        </figure>
      </MethodologySection>

      <MethodologySection
        id="concepts"
        title="Concepts de structure (SMC)"
        intro="Notre lecture s’appuie sur les Smart Money Concepts — une grille de lecture de la structure du prix. Voici les éléments que le moteur détecte et affiche."
      >
        <div className="grid gap-4 sm:grid-cols-2">
          {SMC_CONCEPTS.map((concept) => (
            <ConceptCard key={concept.id} concept={concept} />
          ))}
        </div>
      </MethodologySection>

      <MethodologySection
        id="scores"
        title="Ce que nous décrivons (et comment)"
        intro="Certains éléments de la lecture sont qualifiés (importance d’une zone, statut, phase, incertitude). Voici ce que chacun décrit et les variables qui le composent — aucun n’est une probabilité de réussite."
      >
        <div className="space-y-8">
          {SCORE_FORMULAS.map((formula) => (
            <ScoreFormula key={formula.id} formula={formula} />
          ))}
        </div>
      </MethodologySection>

      <MethodologySection
        id="donnees"
        title="Notre source de données"
        intro={DATA_SOURCE.detail}
      >
        <dl className="grid gap-4 sm:grid-cols-3">
          <div>
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/80">
              Fournisseur
            </dt>
            <dd className="mt-1 text-sm text-foreground">
              {DATA_SOURCE.provider}
            </dd>
          </div>
          <div>
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/80">
              Couverture V1
            </dt>
            <dd className="mt-1 text-sm text-foreground">
              {DATA_SOURCE.coverage}
            </dd>
          </div>
          <div>
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/80">
              Fréquence
            </dt>
            <dd className="mt-1 text-sm text-foreground">
              {DATA_SOURCE.refresh}
            </dd>
          </div>
        </dl>
      </MethodologySection>

      <MethodologySection
        id="limites"
        title="Ce que nous ne faisons pas"
        intro="La transparence vaut aussi pour nos limites. Volontairement, l’indicateur ne fait rien de ce qui suit."
      >
        <ul className="space-y-2 text-sm text-muted-foreground">
          {NEVER_DO.map((item) => (
            <li key={item} className="flex items-start gap-2">
              <span
                className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sentinel-bear"
                aria-hidden
              />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      </MethodologySection>

      <MethodologySection
        id="attributions"
        title="Attributions"
        intro="Le graphique en chandeliers de l’espace de lecture s’appuie sur une bibliothèque open-source, créditée ici comme l’exige sa licence."
      >
        <div className="space-y-2 text-sm text-muted-foreground">
          <p>
            <span className="text-foreground">Lightweight Charts™</span> —
            © TradingView, Inc., distribué sous licence{' '}
            <a
              href="https://www.apache.org/licenses/LICENSE-2.0"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-4 hover:text-foreground"
            >
              Apache 2.0
            </a>
            . Bibliothèque d’affichage de graphiques utilisée pour le rendu des
            chandeliers et des zones.
          </p>
          <p className="text-xs">
            À noter : il s’agit d’une bibliothèque <em>d’affichage</em>{' '}
            uniquement. MIA Markets n’utilise aucune API ni aucun flux de données
            de marché de TradingView — les cours affichés proviennent de notre
            propre moteur.
          </p>
        </div>
      </MethodologySection>
    </div>
  );
}
