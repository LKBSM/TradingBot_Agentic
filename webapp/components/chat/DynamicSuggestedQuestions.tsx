'use client';

/**
 * DG-114-REDUCED — 3 questions suggérées contextuelles dynamiques.
 *
 * Dérivées en live du signal courant (conviction tier, top composante,
 * event ≤ 4 h / IC / régime). Diffère du composant SuggestedQuestions
 * scripté : ici un clic envoie la question à ``askFreeForm`` (route
 * LLM réelle) au lieu de l'``appendExchange`` scripté. La réponse
 * passera par la machinerie compliance complète (DG-112 gates + DG-042
 * tier-routing).
 */
import * as React from 'react';
import { cn } from '@/lib/utils';
import { suggestedQuestions } from '@/lib/chat/suggested-questions';
import type { InsightSignalV2 } from '@/types/insight';

export interface DynamicSuggestedQuestionsProps {
  signal: InsightSignalV2;
  language?: 'fr' | 'en';
  onAsk(question: string): void;
  disabled?: boolean;
}

export function DynamicSuggestedQuestions({
  signal,
  language = 'fr',
  onAsk,
  disabled = false,
}: DynamicSuggestedQuestionsProps) {
  const questions = React.useMemo(
    () => suggestedQuestions(signal, language),
    [signal, language],
  );

  return (
    <div className="space-y-2" data-testid="dynamic-suggested-questions">
      <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        Questions contextuelles
      </p>
      <div className="flex flex-col gap-2">
        {questions.map((q) => (
          <button
            key={q.source}
            type="button"
            disabled={disabled}
            data-source={q.source}
            onClick={() => onAsk(q.text)}
            className={cn(
              'rounded-lg border border-dashed border-border bg-background px-3 py-2 text-left text-sm',
              'transition-colors hover:bg-accent hover:text-accent-foreground',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
              'disabled:cursor-not-allowed disabled:opacity-50',
            )}
          >
            {q.text}
          </button>
        ))}
      </div>
    </div>
  );
}
