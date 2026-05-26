'use client';

import { SendHorizonal } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

/**
 * Visually-present but disabled free-text input. The free-form chat will land
 * after the backend integration sprint (and proper LLM cost controls). For
 * V1 we keep the affordance discoverable but inert, with a tooltip that
 * sets expectations cleanly.
 */
export function ChatInputStub({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        'flex items-center gap-2 rounded-xl border border-dashed border-border bg-muted/30 p-2',
        className,
      )}
    >
      <Tooltip>
        <TooltipTrigger asChild>
          <input
            type="text"
            disabled
            aria-disabled="true"
            placeholder="Saisie libre — disponible bientôt"
            className="flex-1 bg-transparent px-2 py-1 text-sm text-muted-foreground placeholder:text-muted-foreground/70 focus:outline-none"
          />
        </TooltipTrigger>
        <TooltipContent side="top">
          La saisie libre arrive après l'intégration backend (V2). Pour le
          moment, utilise les questions suggérées ci-dessus.
        </TooltipContent>
      </Tooltip>
      <button
        type="button"
        disabled
        aria-disabled="true"
        aria-label="Envoyer (désactivé en V1)"
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground"
      >
        <SendHorizonal className="h-4 w-4" />
      </button>
    </div>
  );
}
