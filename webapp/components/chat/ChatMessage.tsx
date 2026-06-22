'use client';

import { Check, Copy, Info, User } from 'lucide-react';
import * as React from 'react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { renderMarkdown } from '@/lib/chat/markdown';
import { cn } from '@/lib/utils';
import { AgentAvatar } from './AgentAvatar';

interface ChatMessageProps {
  role: 'user' | 'assistant';
  text: string;
  /**
   * Set when a niveau-1.5 defence layer redirected this answer. Renders a
   * discreet info badge with a pedagogical tooltip — never an alarming banner.
   */
  blockedReason?: string | null;
  /**
   * Display-only: the assistant turn carried validated chart view actions, so
   * the workspace updated the chart. We surface a discreet "Vue mise à jour"
   * confirmation under the bubble. Never affects the agent or view-control
   * logic — it only mirrors what already happened.
   */
  viewUpdated?: boolean;
}

/** Niveau 1.5 strict, non-anxiogène — same wording validated in T3. */
const REDIRECT_TOOLTIP =
  'MIA Markets décrit les conditions de marché. Pour les questions d’action ou de conseil, c’est à vous de décider selon vos propres critères.';

/**
 * Single chat exchange row. The assistant message is full-width with the brand
 * avatar + name and a discreet copy action; the user message sits on the right
 * in an accent bubble. Assistant replies render light Markdown; user input
 * stays verbatim. New rows fade in softly (`chat-msg-in`).
 */
export function ChatMessage({
  role,
  text,
  blockedReason,
  viewUpdated,
}: ChatMessageProps) {
  const isUser = role === 'user';
  const showRedirect = !isUser && Boolean(blockedReason);

  if (isUser) {
    return (
      <div className="chat-msg-in flex w-full justify-end gap-2.5">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-tr-sm border border-[hsl(35_92%_55%/0.30)] bg-[hsl(35_92%_55%/0.14)] px-3.5 py-2.5 text-sm leading-relaxed text-foreground">
          {text}
        </div>
        <div
          aria-hidden
          className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-secondary text-secondary-foreground"
        >
          <User className="h-4 w-4" />
        </div>
      </div>
    );
  }

  return (
    <div className="chat-msg-in flex w-full gap-2.5" role="status">
      <AgentAvatar size="sm" className="mt-0.5" />
      <div className="group flex min-w-0 flex-col gap-1">
        <span className="px-0.5 text-xs font-medium text-muted-foreground">
          M.I.A Agent
        </span>
        <div className="max-w-[88%] rounded-2xl rounded-tl-sm border border-border bg-muted/60 px-3.5 py-2.5 text-sm leading-relaxed text-foreground">
          {renderMarkdown(text)}
        </div>

        <div className="flex items-center gap-2 pl-0.5">
          <CopyButton text={text} />
          {viewUpdated && (
            <span className="inline-flex items-center gap-1 rounded-full border border-[hsl(var(--sentinel-bull)/0.3)] bg-[hsl(var(--sentinel-bull)/0.12)] px-2 py-0.5 text-[11px] text-[hsl(var(--sentinel-bull))]">
              <Check className="h-3 w-3" aria-hidden />
              Vue mise à jour
            </span>
          )}
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
      </div>
    </div>
  );
}

/** Discreet copy-to-clipboard for an assistant reply (visible on row hover). */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = React.useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard blocked (insecure context / permissions) — fail silently.
    }
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      aria-label={copied ? 'Réponse copiée' : 'Copier la réponse'}
      className={cn(
        'flex items-center gap-1 rounded px-1 py-0.5 text-[11px] text-muted-foreground',
        'opacity-0 transition-opacity hover:text-foreground focus-visible:opacity-100',
        'group-hover:opacity-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
      )}
    >
      {copied ? (
        <>
          <Check className="h-3 w-3" aria-hidden />
          Copié
        </>
      ) : (
        <>
          <Copy className="h-3 w-3" aria-hidden />
          Copier
        </>
      )}
    </button>
  );
}
