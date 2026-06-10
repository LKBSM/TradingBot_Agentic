import { Bot, Info, User } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { renderMarkdown } from '@/lib/chat/markdown';
import { cn } from '@/lib/utils';

interface ChatMessageProps {
  role: 'user' | 'assistant';
  text: string;
  /**
   * Set when a niveau-1.5 defence layer redirected this answer. Renders a
   * discreet info badge with a pedagogical tooltip — never an alarming banner.
   */
  blockedReason?: string | null;
}

/** Niveau 1.5 strict, non-anxiogène — same wording validated in T3. */
const REDIRECT_TOOLTIP =
  'MIA Markets décrit les conditions de marché. Pour les questions d’action ou de conseil, c’est à vous de décider selon vos propres critères.';

/**
 * Single chat bubble. User on the right (secondary), assistant on the left
 * (muted card style). Long-form assistant replies wrap and preserve newlines.
 * When `blockedReason` is set, a small info badge sits under the bubble.
 */
export function ChatMessage({ role, text, blockedReason }: ChatMessageProps) {
  const isUser = role === 'user';
  const showRedirect = !isUser && Boolean(blockedReason);
  return (
    <div
      className={cn(
        'flex w-full gap-2',
        isUser ? 'justify-end' : 'justify-start',
      )}
      role={role === 'assistant' ? 'status' : undefined}
    >
      {!isUser && (
        <div
          aria-hidden
          className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground"
        >
          <Bot className="h-4 w-4" />
        </div>
      )}
      <div className="flex max-w-[85%] flex-col gap-1">
        <div
          className={cn(
            'rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed',
            isUser
              ? 'whitespace-pre-wrap rounded-tr-sm bg-primary text-primary-foreground'
              : 'rounded-tl-sm bg-muted text-foreground',
          )}
        >
          {/* Assistant replies may contain light Markdown (**bold**, lists) —
              render it cleanly. User input stays verbatim (pre-wrap). */}
          {isUser ? text : renderMarkdown(text)}
        </div>
        {showRedirect && (
          <TooltipProvider delayDuration={150}>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="flex w-fit items-center gap-1 rounded px-1 text-[11px] text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  aria-label="Pourquoi cette réponse a été recadrée"
                >
                  <Info className="h-3 w-3" aria-hidden />
                  Question recadrée
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="max-w-[260px] text-xs">
                {REDIRECT_TOOLTIP}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>
      {isUser && (
        <div
          aria-hidden
          className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-secondary text-secondary-foreground"
        >
          <User className="h-4 w-4" />
        </div>
      )}
    </div>
  );
}
