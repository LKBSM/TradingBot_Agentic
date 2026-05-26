'use client';

import { Bot, MessageCircle, RotateCcw } from 'lucide-react';
import * as React from 'react';
import { Button } from '@/components/ui/button';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { ChatInputStub } from './ChatInputStub';
import { ChatMessage } from './ChatMessage';
import { useChat } from './ChatProvider';
import { SuggestedQuestions } from './SuggestedQuestions';
import { FALLBACK_SCRIPT, getChatbotScript } from '@/lib/chatbot';
import {
  formatInstrument,
  formatTimeframe,
} from '@/lib/insight-formatters';
import type { ChatbotQuestion } from '@/types/chatbot';

/**
 * Slide-over chat panel — desktop ≥ 768px keeps the underlying card visible
 * (right-side sheet, ~480px wide). On mobile the sheet becomes a fullscreen
 * surface so the conversation has room to breathe.
 *
 * Rendered once at the layout level — opened/closed via the shared
 * ChatProvider context.
 */
export function ChatPanel() {
  const {
    isOpen,
    activeSignal,
    turns,
    close,
    appendExchange,
    resetTurns,
  } = useChat();
  const scrollRef = React.useRef<HTMLDivElement>(null);

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

  // Auto-scroll to the bottom whenever a new exchange lands.
  React.useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [turns.length]);

  function handlePick(q: ChatbotQuestion) {
    appendExchange({ questionId: q.id, text: q.text, reply: q.reply });
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
              <Bot className="h-4 w-4" aria-hidden />
            </div>
            <div className="space-y-0.5">
              <SheetTitle className="text-base">Sentinel</SheetTitle>
              <SheetDescription
                id="chat-panel-description"
                className="text-xs"
              >
                {activeSignal
                  ? `${formatInstrument(activeSignal)} · ${formatTimeframe(activeSignal)} · contexte injecté`
                  : 'Sélectionne une lecture pour ouvrir le contexte.'}
              </SheetDescription>
            </div>
          </div>
        </SheetHeader>

        <div
          ref={scrollRef}
          className="flex-1 space-y-4 overflow-y-auto px-5 py-4"
        >
          <IntroBubble script={script} />

          {turns.map((t) => (
            <ChatMessage key={t.id} role={t.role} text={t.text} />
          ))}

          {turns.length > 0 && (
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

        <div className="border-t bg-background px-5 py-4">
          <SuggestedQuestions
            questions={script.questions}
            consumedIds={consumedIds}
            onPick={handlePick}
          />
          <div className="mt-3">
            <ChatInputStub />
          </div>
          {/* LEGAL-PENDING: standalone compliance line at the bottom of the
              chat panel — to be aligned with the legal terminal wording on
              educational-use posture (UE 2024/2811). */}
          <p className="mt-3 text-[11px] italic text-muted-foreground">
            Sentinel répond à des questions sur la lecture algorithmique. Il
            ne donne ni signal de trading, ni recommandation personnalisée.
          </p>
        </div>
      </SheetContent>
    </Sheet>
  );
}

function IntroBubble({
  script,
}: {
  script: { instrument_label: string; questions: ReadonlyArray<ChatbotQuestion> };
}) {
  return (
    <div className="rounded-2xl rounded-tl-sm bg-muted px-3.5 py-2.5 text-sm leading-relaxed">
      <p className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        <MessageCircle className="h-3 w-3" aria-hidden />
        Sentinel · {script.instrument_label}
      </p>
      <p className="mt-2">
        Pose-moi une question sur cette lecture. Je peux décomposer la
        conviction, expliquer un terme, contextualiser un événement, ou
        comparer aux setups historiques similaires.
      </p>
      <p className="mt-2 text-xs italic text-muted-foreground">
        Je ne donne ni signal d'achat ou de vente, ni conseil en investissement.
      </p>
    </div>
  );
}
