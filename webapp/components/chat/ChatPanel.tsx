'use client';

import { RotateCcw } from 'lucide-react';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import { Button } from '@/components/ui/button';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { AgentAvatar } from './AgentAvatar';
import { ChatInput } from './ChatInput';
import { ChatMessage } from './ChatMessage';
import { useChat } from './ChatProvider';
// Keep main's anchor-scroll UX (anti scroll-to-bottom) inside the redesign's UI.
import { useChatAnchorScroll } from './useChatAnchorScroll';
import { SuggestedQuestions } from './SuggestedQuestions';
import { ThinkingIndicator } from './ThinkingIndicator';
import { useChatbotScriptOrFallback } from '@/lib/chatbot';
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
  const t = useTranslations('chat');
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

  const script = useChatbotScriptOrFallback(activeSignal?.id ?? null);

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
        <SheetHeader className="space-y-0 border-b px-5 py-4 text-left">
          <div className="flex items-center gap-3">
            <AgentAvatar size="md" />
            <div className="min-w-0 flex-1 space-y-0.5">
              <SheetTitle className="flex items-center gap-1.5 text-base">
                M.I.A Agent
                <span
                  className="h-1.5 w-1.5 rounded-full bg-[hsl(var(--sentinel-bull))]"
                  aria-hidden
                />
              </SheetTitle>
              <SheetDescription
                id="chat-panel-description"
                className="text-xs"
              >
                {activeSignal
                  ? t('panelContext', {
                      instrument: formatInstrument(activeSignal.instrument),
                      timeframe: formatTimeframe(activeSignal.timeframe),
                    })
                  : t('panelNoContext')}
              </SheetDescription>
              <p className="text-[10.5px] italic text-muted-foreground/85">
                {t('panelTagline')}
              </p>
            </div>
          </div>
        </SheetHeader>

        <div
          ref={scrollRef}
          className="flex flex-1 flex-col gap-4 overflow-y-auto px-5 py-4"
        >
          <IntroBubble script={script} apiAvailable={apiAvailable} />

          {turns.map((t) => (
            <ChatMessage
              key={t.id}
              role={t.role}
              text={t.text}
              blockedReason={t.blockedReason}
              viewUpdated={t.viewUpdated}
            />
          ))}

          {isLoading && <ThinkingIndicator />}

          {turns.length > 0 && !isLoading && (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={resetTurns}
              className="mx-auto flex h-7 items-center gap-1 text-xs text-muted-foreground"
            >
              <RotateCcw className="h-3 w-3" aria-hidden />
              {t('reset')}
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
              educational-use posture. */}
          <p className="text-center text-[10.5px] italic text-muted-foreground/70">
            {t('complianceLine')}
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
  const t = useTranslations('chat');
  return (
    <div className="chat-msg-in flex w-full gap-2.5">
      <AgentAvatar size="sm" className="mt-0.5" />
      <div className="flex min-w-0 flex-col gap-1">
        <span className="px-0.5 text-xs font-medium text-muted-foreground">
          {t('introHeader', { instrument: script.instrument_label })}
        </span>
        <div className="max-w-[88%] rounded-2xl rounded-tl-sm border border-border bg-muted/60 px-3.5 py-2.5 text-sm leading-relaxed">
          <p>{t('introBody')}</p>
          <p className="mt-2 text-xs italic text-muted-foreground">
            {apiAvailable === false ? t('introOffline') : t('introDisclaimer')}
          </p>
        </div>
      </div>
    </div>
  );
}
