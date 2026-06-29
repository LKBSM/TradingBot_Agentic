'use client';

import { Loader2, MessageCircle, RotateCcw } from 'lucide-react';
import * as React from 'react';
import { Button } from '@/components/ui/button';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { ChatInput } from './ChatInput';
import { ChatMessage } from './ChatMessage';
import { useChat } from './ChatProvider';
import { useChatAnchorScroll } from './useChatAnchorScroll';
import { MiaAgentLogo } from './MiaAgentLogo';
import { SuggestedQuestions } from './SuggestedQuestions';
import { FALLBACK_SCRIPT, getChatbotScript } from '@/lib/chatbot';
import {
  formatInstrument,
  formatTimeframe,
} from '@/lib/market-reading/formatters';
import type { ChatbotQuestion } from '@/types/chatbot';

/**
 * Slide-over chat panel — desktop ≥ 768px keeps the underlying card visible
 * (right-side sheet, ~480-560px wide depending on viewport). On mobile the
 * sheet becomes a fullscreen surface so the conversation has room to breathe.
 *
 * Rendered once at the layout level — opened/closed via the shared
 * ChatProvider context. Suggested questions hit the scripted fallback so
 * answers stay deterministic and free. Free-text input goes through
 * POST /api/chatbot/message (FastAPI backend, 3 niveau-1.5 defence layers;
 * friendly fallback when the backend is unavailable).
 */
export function ChatPanel() {
  const {
    isOpen,
    activeSignal,
    turns,
    isLoading,
    apiAvailable,
    close,
    appendExchange,
    resetTurns,
  } = useChat();
  // Anchor the latest question near the top after sending instead of jumping to
  // the bottom of a long reply (UX only — no chat logic changes).
  const scrollRef = useChatAnchorScroll(turns, isLoading);

  const script = React.useMemo(() => {
    if (!activeSignal) return FALLBACK_SCRIPT;
    return getChatbotScript(activeSignal.id) ?? FALLBACK_SCRIPT;
  }, [activeSignal]);

  const consumedIds = React.useMemo(
    () =>
      new Set(
        turns
          .filter((t) => t.role === 'user')
          .map((t) => t.id.split('-q-')[0])
          .filter((id): id is string => Boolean(id)),
      ),
    [turns],
  );

  function handlePick(q: ChatbotQuestion) {
    appendExchange({
      questionId: q.id,
      text: q.text,
      reply: q.reply,
      source: 'scripted',
    });
  }

  return (
    <Sheet open={isOpen} onOpenChange={(next) => !next && close()}>
      <SheetContent
        side="right"
        className="flex w-full flex-col gap-0 p-0 sm:max-w-md md:max-w-lg"
        aria-describedby="chat-panel-description"
      >
        <SheetHeader className="border-b px-5 py-4 text-left">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground">
              <MiaAgentLogo className="h-5 w-5" />
            </div>
            <div className="space-y-0.5">
              <SheetTitle className="text-base">M.I.A Agent</SheetTitle>
              <SheetDescription
                id="chat-panel-description"
                className="text-xs"
              >
                {activeSignal
                  ? `${formatInstrument(activeSignal.instrument)} · ${formatTimeframe(activeSignal.timeframe)} · contexte injecté`
                  : 'Sélectionne une lecture pour ouvrir le contexte.'}
              </SheetDescription>
            </div>
          </div>
        </SheetHeader>

        <div
          ref={scrollRef}
          className="flex-1 space-y-4 overflow-y-auto px-5 py-4"
        >
          <IntroBubble script={script} apiAvailable={apiAvailable} />

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

        <div className="space-y-3 border-t bg-background px-5 py-4">
          <SuggestedQuestions
            questions={script.questions}
            consumedIds={consumedIds}
            onPick={handlePick}
          />
          <ChatInput />
          {/* LEGAL-PENDING: standalone compliance line at the bottom of the
              chat panel — aligned with legal terminal wording on
              educational-use posture (UE 2024/2811 + MiFID II 03/2026). */}
          <p className="text-[11px] italic text-muted-foreground">
            M.I.A Agent répond à des questions sur la lecture algorithmique.
            Il ne donne ni signal de trading, ni recommandation personnalisée.
          </p>
        </div>
      </SheetContent>
    </Sheet>
  );
}

function IntroBubble({
  script,
  apiAvailable,
}: {
  script: { instrument_label: string; questions: ReadonlyArray<ChatbotQuestion> };
  apiAvailable: boolean | 'unknown';
}) {
  return (
    <div className="rounded-2xl rounded-tl-sm bg-muted px-3.5 py-2.5 text-sm leading-relaxed">
      <p className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        <MessageCircle className="h-3 w-3" aria-hidden />
        M.I.A Agent · {script.instrument_label}
      </p>
      <p className="mt-2">
        Pose-moi une question sur cette lecture. Je peux t&apos;expliquer
        ce qu&apos;elle décrit, vulgariser un terme technique, ou
        contextualiser un événement à venir.
      </p>
      <p className="mt-2 text-xs italic text-muted-foreground">
        {apiAvailable === false ? (
          <>
            Mode scripted : utilise les suggestions ci-dessous. La saisie
            libre nécessite la clef Anthropic côté serveur.
          </>
        ) : (
          <>
            Je ne donne ni signal d&apos;achat ou de vente, ni conseil en
            investissement.
          </>
        )}
      </p>
    </div>
  );
}
