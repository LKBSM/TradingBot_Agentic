import * as React from 'react';
import { cn } from '@/lib/utils';
import { AgentAvatar } from './AgentAvatar';

export interface WelcomeSuggestion {
  id: string;
  text: string;
  icon?: React.ReactNode;
}

interface ChatWelcomeProps {
  title: string;
  subtitle: string;
  /** On-brand starter questions. Empty array → hero only, no chips. */
  suggestions?: ReadonlyArray<WelcomeSuggestion>;
  onPick?(suggestion: WelcomeSuggestion): void;
  /** Optional footnote under the suggestions (e.g. offline / honesty note). */
  note?: React.ReactNode;
  className?: string;
}

/**
 * Empty-state hero for the chat surface — brand avatar, a warm prompt, and a
 * few clickable on-brand starter questions. The big UX win: the panel never
 * opens blank. Presentational only; clicking a suggestion just calls back with
 * its text (the caller decides whether that hits the scripted flow or the LLM).
 */
export function ChatWelcome({
  title,
  subtitle,
  suggestions = [],
  onPick,
  note,
  className,
}: ChatWelcomeProps) {
  return (
    <div
      className={cn(
        'chat-msg-in flex flex-1 flex-col items-center justify-center gap-3.5 px-2 py-8 text-center',
        className,
      )}
    >
      <AgentAvatar size="lg" />
      <h2 className="text-base font-semibold text-foreground">{title}</h2>
      <p className="max-w-[18rem] text-[13px] leading-relaxed text-muted-foreground">
        {subtitle}
      </p>

      {suggestions.length > 0 && (
        // MIA-1: compact wrapping chips (1–2 rows), not full-width stacked
        // blocks — they start a conversation, they don't crowd the panel. They
        // vanish entirely once the first exchange lands (empty-state only).
        <div className="mt-1 flex flex-wrap justify-center gap-1.5" data-testid="chat-starters">
          {suggestions.map((s) => (
            <button
              key={s.id}
              type="button"
              data-testid="chat-starter"
              onClick={() => onPick?.(s)}
              className={cn(
                'inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/40 px-3 py-1.5 text-left text-xs text-foreground',
                'transition-colors hover:border-[hsl(35_92%_55%/0.4)] hover:bg-muted',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
              )}
            >
              {s.icon && (
                <span className="shrink-0 text-[hsl(var(--sentinel-warn))]">
                  {s.icon}
                </span>
              )}
              {s.text}
            </button>
          ))}
        </div>
      )}

      {note && (
        <p className="mt-1 max-w-[18rem] text-[11px] italic text-muted-foreground">
          {note}
        </p>
      )}
    </div>
  );
}
