import Link from 'next/link';
import { HelpCircle, ArrowRight } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';

interface FaqEntry {
  id: string;
  question: string;
  answer: React.ReactNode;
}

/**
 * Section L5.2 — FAQ.
 *
 * 6 questions choisies sur les vrais points d'objection :
 *  1. Est-ce un service de signaux ? (positionnement)
 *  2. Promesse de performance ? (positionnement descriptif + lien Honnêteté)
 *  3. Comment fonctionne le chatbot ? (Claude + contexte)
 *  4. Suis-je éligible ? (âge + consentement)
 *  5. Annulation
 *  6. Mes données, mon historique
 *
 * Règle éditoriale (nettoyage claims 2026-07-04) : aucune référence
 * réglementaire non vérifiée, aucun périmètre géographique annoncé, aucune
 * promesse (remboursement, export…) non implémentée.
 *
 * Accordion Radix — déjà installé, accessible clavier/SR de base.
 */
const FAQ: ReadonlyArray<FaqEntry> = [
  {
    id: 'q1-signaux',
    question: 'MIA est-il un service de signaux de trading ?',
    answer: (
      <>
        Non. MIA Markets produit des <strong>lectures de marché
        descriptives</strong> (structure, régime, événements, lecture narrée) sur un
        actif et une unité de temps. Le chatbot refuse explicitement les
        questions de type «&nbsp;dois-je acheter ?&nbsp;». Aucune
        recommandation personnalisée n&apos;est délivrée.
      </>
    ),
  },
  {
    id: 'q2-precision',
    question: 'MIA promet-elle une performance ou un gain ?',
    answer: (
      <>
        Non. MIA est un <strong>outil de compréhension augmentée</strong>,
        pas un moteur de performance : elle <em>décrit</em> le marché
        (structure, régime, événements) sans promettre de rendement ni se
        présenter comme un système de trading. Ce que nous mesurons et
        assumons publiquement, c&apos;est la <strong>fidélité descriptive</strong>{' '}
        de nos lectures — détaillée dans la section{' '}
        <a
          href="#honnetete"
          className="underline-offset-2 hover:underline"
        >
          Honnêteté
        </a>
        .
      </>
    ),
  },
  {
    id: 'q3-chatbot',
    question: 'Comment fonctionne le chatbot M.I.A Agent ?',
    answer: (
      <>
        M.I.A Agent est l&apos;assistant conversationnel de MIA Markets. Il
        utilise <strong>Claude (Anthropic)</strong> avec le contexte de
        la lecture en cours injecté en système : structure, régime,
        événements macro à venir, instrument et unité de temps. Le prompt
        système lui interdit toute recommandation personnalisée. Si la clé
        API est absente, un fallback scripté pédagogique prend le relais —
        aucune réponse hallucinée ne sera servie.
      </>
    ),
  },
  {
    id: 'q4-eligibilite',
    question: 'Suis-je éligible à MIA Markets ?',
    answer: (
      <>
        Vous devez avoir <strong>18 ans minimum</strong>, accepter les{' '}
        <a
          href="/conditions"
          className="underline-offset-2 hover:underline"
        >
          conditions d&apos;utilisation
        </a>{' '}
        et confirmer que vous comprenez le caractère non personnalisé de
        l&apos;analyse. Les restrictions applicables selon votre
        juridiction sont détaillées dans les conditions.
      </>
    ),
  },
  {
    id: 'q5-annulation',
    question: 'Comment fonctionne l’annulation ?',
    answer: (
      <>
        <strong>Annulation en un clic</strong> depuis votre espace
        compte — aucune relance commerciale, aucun questionnaire de
        rétention. L&apos;abonnement reste actif jusqu&apos;à la fin de la
        période déjà payée.
      </>
    ),
  },
  {
    id: 'q6-donnees',
    question: 'Que devient mon historique et mes données ?',
    answer: (
      <>
        Vos lectures et vos conversations chatbot restent{' '}
        <strong>privées</strong>. Vous pouvez demander à tout moment
        l&apos;accès, la rectification ou l&apos;effacement de vos données
        (contact@mia.markets). Aucune revente, aucun croisement
        publicitaire. Détail dans la{' '}
        <a
          href="/confidentialite"
          className="underline-offset-2 hover:underline"
        >
          politique de confidentialité
        </a>
        .
      </>
    ),
  },
];

export function FaqSection() {
  return (
    <section
      id="faq"
      aria-labelledby="faq-title"
      className="container-prose py-16 sm:py-20"
    >
      <header className="mb-8 space-y-3">
        <Badge
          variant="secondary"
          className="text-[11px] uppercase tracking-wider"
        >
          <HelpCircle className="mr-1 h-3 w-3" aria-hidden />
          Questions fréquentes
        </Badge>
        <h2
          id="faq-title"
          className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl"
        >
          Vous vous demandez probablement…
        </h2>
      </header>

      <Accordion type="single" collapsible className="w-full">
        {FAQ.map((entry) => (
          <AccordionItem key={entry.id} value={entry.id}>
            <AccordionTrigger className="text-left">
              {entry.question}
            </AccordionTrigger>
            <AccordionContent className="text-muted-foreground">
              {entry.answer}
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>

      <p className="mt-8 text-sm text-muted-foreground">
        Vous voulez creuser le fonctionnement&nbsp;?{' '}
        <Link
          href="/methodology"
          className="inline-flex items-center gap-1 font-medium text-foreground underline-offset-4 hover:underline"
        >
          Comment notre indicateur décrit les structures de marché
          <ArrowRight className="h-3.5 w-3.5" aria-hidden />
        </Link>
      </p>
    </section>
  );
}
