import { InsightGallery } from '@/components/insight/InsightGallery';
import { SAMPLE_SIGNALS } from '@/lib/mocks';

/**
 * Sprint F4 demo — three mocked signals + a chatbot pilier accessible from
 * each card's "Demander à Sentinel" CTA. The proper landing (hero + how-it-
 * works + pricing + footer) is rebuilt in F5.
 */
export default function LandingPage() {
  // Demo aid: open the History section on the first card so the PF + IC95%
  // hero differentiator is visible without interaction.
  const defaultOpenByIndex = new Map<
    number,
    ReadonlyArray<'history'>
  >([[0, ['history']]]);

  return (
    <div className="container-prose space-y-12 py-10 sm:py-16">
      <header className="space-y-3">
        <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Sprint F4 · hero + sections + chatbot pilier
        </p>
        <h1 className="text-balance text-3xl font-semibold tracking-tight sm:text-4xl">
          Smart Sentinel — Lecture de marché
        </h1>
        <p className="max-w-2xl text-pretty text-muted-foreground">
          Trois lectures algorithmiques mockées. Hero permanent + cinq sections
          dépliables + chatbot contextualisé sur chaque lecture (5 questions
          suggérées par signal, incluant un refus pédagogique).
        </p>
      </header>

      <InsightGallery
        signals={SAMPLE_SIGNALS}
        defaultOpenByIndex={defaultOpenByIndex}
      />
    </div>
  );
}
