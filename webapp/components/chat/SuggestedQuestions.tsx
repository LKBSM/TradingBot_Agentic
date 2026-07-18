import { useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import type { ChatbotQuestion } from '@/types/chatbot';

interface SuggestedQuestionsProps {
  questions: readonly ChatbotQuestion[];
  /** Question IDs the user has already asked (chip hidden once answered). */
  consumedIds?: ReadonlySet<string>;
  onPick(question: ChatbotQuestion): void;
}

/**
 * Vertical stack of clickable suggestion chips. Once a question is asked,
 * its chip is removed from the stack so the user never sees the same
 * conversation twice in V1. When no suggestion remains, a soft reminder
 * is rendered pointing at the disabled free-text input.
 */
export function SuggestedQuestions({
  questions,
  consumedIds = new Set(),
  onPick,
}: SuggestedQuestionsProps) {
  const t = useTranslations('chat');
  const remaining = questions.filter((q) => !consumedIds.has(q.id));

  if (remaining.length === 0) {
    return (
      <p className="rounded-md border border-dashed border-border bg-muted/40 px-3 py-2 text-xs italic text-muted-foreground">
        {t('allSuggestionsAsked')}
      </p>
    );
  }

  return (
    <div className="space-y-2">
      <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        {t('suggestedLabel')}
      </p>
      <div className="flex flex-col gap-2">
        {remaining.map((q) => (
          <button
            key={q.id}
            type="button"
            onClick={() => onPick(q)}
            className={cn(
              'rounded-xl border border-border bg-muted/40 px-3.5 py-2.5 text-left text-sm',
              'transition-all hover:-translate-y-0.5 hover:border-[hsl(35_92%_55%/0.4)] hover:bg-muted',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            )}
          >
            {q.text}
          </button>
        ))}
      </div>
    </div>
  );
}
