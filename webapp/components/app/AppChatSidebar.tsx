'use client';

import { Loader2, RotateCcw } from 'lucide-react';
import * as React from 'react';
import { ChatInput } from '@/components/chat/ChatInput';
import { ChatMessage } from '@/components/chat/ChatMessage';
import { MiaAgentLogo } from '@/components/chat/MiaAgentLogo';
import { useChat } from '@/components/chat/ChatProvider';
import { Button } from '@/components/ui/button';
import {
  formatInstrument,
  formatTimeframe,
} from '@/lib/market-reading/formatters';
import type { Combo } from '@/lib/market-reading/store';

/**
 * Right column — the permanently docked Sentinel chat. Unlike the layout's
 * slide-over Sheet (used on the landing), this renders inline and is always
 * visible on /app. It shares the same ChatProvider context; the active combo's
 * context is bound via `openForCombo` upstream (no Sheet, no modal).
 */
export function AppChatSidebar({ active }: { active: Combo | null }) {
  const { turns, isLoading, apiAvailable, resetTurns } = useChat();
  const scrollRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [turns.length, isLoading]);

  return (
    <aside
      aria-label="Assistant M.I.A Agent"
      className="flex h-full min-h-0 flex-col rounded-lg border border-border/60 bg-card"
    >
      <header className="flex items-center gap-2 border-b border-border/60 px-4 py-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <MiaAgentLogo className="h-5 w-5" />
        </div>
        <div className="min-w-0">
          <p className="text-sm font-semibold leading-tight">M.I.A Agent</p>
          <p className="truncate text-xs text-muted-foreground">
            {active
              ? `${formatInstrument(active.instrument)} · ${formatTimeframe(active.timeframe)}`
              : 'Sélectionnez une combinaison pour discuter de sa lecture.'}
          </p>
        </div>
      </header>

      <div
        ref={scrollRef}
        className="flex-1 space-y-4 overflow-y-auto px-4 py-4"
      >
        <div className="rounded-2xl rounded-tl-sm bg-muted px-3.5 py-2.5 text-sm leading-relaxed">
          <p className="mt-0">
            {active
              ? 'Pose une question sur cette lecture : décomposer la structure, expliquer un terme, contextualiser un événement.'
              : 'Choisis un marché à gauche, puis pose-moi une question sur sa lecture.'}
          </p>
          <p className="mt-2 text-xs italic text-muted-foreground">
            {apiAvailable === false
              ? 'Mode hors-ligne : la saisie libre nécessite la clef Anthropic côté serveur.'
              : 'Je ne donne ni signal d’achat ou de vente, ni conseil en investissement.'}
          </p>
        </div>

        {turns.map((t) => (
          <ChatMessage
            key={t.id}
            role={t.role}
            text={t.text}
            blockedReason={t.blockedReason}
          />
        ))}

        {isLoading && (
          <div
            className="flex items-center gap-2 px-1 text-sm text-muted-foreground"
            role="status"
            aria-live="polite"
          >
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            M.I.A Agent réfléchit…
          </div>
        )}

        {turns.length > 0 && !isLoading && (
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={resetTurns}
            className="mx-auto flex h-7 items-center gap-1 text-xs text-muted-foreground"
          >
            <RotateCcw className="h-3 w-3" aria-hidden />
            Réinitialiser la conversation
          </Button>
        )}
      </div>

      <div className="space-y-3 border-t border-border/60 bg-background px-4 py-3">
        <ChatInput />
        {/* LEGAL-PENDING: chat compliance line — aligned with the legal terminal
            wording on educational-use posture (UE 2024/2811 + MiFID II 03/2026). */}
        <p className="text-[11px] italic text-muted-foreground">
          M.I.A Agent répond à des questions sur la lecture algorithmique. Il
          ne donne ni signal de trading, ni recommandation personnalisée.
        </p>
      </div>
    </aside>
  );
}
