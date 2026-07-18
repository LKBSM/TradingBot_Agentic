import Link from 'next/link';
import { useTranslations } from 'next-intl';
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
  q: string;
  a: string;
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
  { id: 'q1-signaux', q: 'q1', a: 'a1' },
  { id: 'q2-precision', q: 'q2', a: 'a2' },
  { id: 'q3-chatbot', q: 'q3', a: 'a3' },
  { id: 'q4-eligibilite', q: 'q4', a: 'a4' },
  { id: 'q5-annulation', q: 'q5', a: 'a5' },
  { id: 'q6-donnees', q: 'q6', a: 'a6' },
];

export function FaqSection() {
  const t = useTranslations('landing.faq');

  const answerElements: Record<string, React.ReactNode> = {
    a1: t.rich('a1', {
      strong: (chunks) => <strong>{chunks}</strong>,
    }),
    a2: t.rich('a2', {
      strong: (chunks) => <strong>{chunks}</strong>,
      em: (chunks) => <em>{chunks}</em>,
      link: (chunks) => (
        <a href="#honnetete" className="underline-offset-2 hover:underline">
          {chunks}
        </a>
      ),
    }),
    a3: t.rich('a3', {
      strong: (chunks) => <strong>{chunks}</strong>,
    }),
    a4: t.rich('a4', {
      strong: (chunks) => <strong>{chunks}</strong>,
      link: (chunks) => (
        <Link
          href="/conditions"
          className="underline-offset-2 hover:underline"
        >
          {chunks}
        </Link>
      ),
    }),
    a5: t.rich('a5', {
      strong: (chunks) => <strong>{chunks}</strong>,
    }),
    a6: t.rich('a6', {
      strong: (chunks) => <strong>{chunks}</strong>,
      link: (chunks) => (
        <Link
          href="/confidentialite"
          className="underline-offset-2 hover:underline"
        >
          {chunks}
        </Link>
      ),
    }),
  };

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
          {t('kicker')}
        </Badge>
        <h2
          id="faq-title"
          className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl"
        >
          {t('title')}
        </h2>
      </header>

      <Accordion type="single" collapsible className="w-full">
        {FAQ.map((entry) => (
          <AccordionItem key={entry.id} value={entry.id}>
            <AccordionTrigger className="text-left">
              {t(entry.q)}
            </AccordionTrigger>
            <AccordionContent className="text-muted-foreground">
              {answerElements[entry.a]}
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>

      <p className="mt-8 text-sm text-muted-foreground">
        {t('methodologyPrompt')}{' '}
        <Link
          href="/methodology"
          className="inline-flex items-center gap-1 font-medium text-foreground underline-offset-4 hover:underline"
        >
          {t('methodologyLink')}
          <ArrowRight className="h-3.5 w-3.5" aria-hidden />
        </Link>
      </p>
    </section>
  );
}
