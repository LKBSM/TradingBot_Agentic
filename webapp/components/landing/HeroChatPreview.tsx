'use client';

import { Bot, MessageCircle } from 'lucide-react';
import * as React from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { useChat } from '@/components/chat/ChatProvider';
import { getChatbotScript } from '@/lib/chatbot';
import { cn } from '@/lib/utils';
import type { InsightSignalV2 } from '@/types/insight';

interface HeroChatPreviewProps {
  signal: InsightSignalV2;
  /** Delay (ms) before the intro bubble appears — synced with card composition. */
  introDelayMs?: number;
}

/**
 * Hero-only chat preview — rendered INLINE alongside the MarketReadingCard
 * on the desktop hero. Visually replicates the panel surface (intro
 * bubble + 3 suggested questions) without mounting the Sheet slide-over.
 *
 * Clicking a question OR the "Continuer dans le chat" CTA opens the real
 * <ChatPanel /> (Sheet) via the shared ChatProvider, with the same signal
 * context pre-injected. Effect : the visitor sees the conversation start
 * inline, then expands to a full panel for real interaction.
 */
export function HeroChatPreview({
  signal,
  introDelayMs = 1400,
}: HeroChatPreviewProps) {
  const { openFor, appendExchange } = useChat();
  const [showIntro, setShowIntro] = React.useState(false);

  const script = React.useMemo(() => getChatbotScript(signal.id), [signal.id]);
  const previewQuestions = React.useMemo(
    () => script?.questions.slice(0, 3) ?? [],
    [script],
  );

  React.useEffect(() => {
    // Delay the intro bubble so the card finishes its composition first,
    // creating the impression that Sentinel "just finished reading" the
    // signal before greeting.
    const t = window.setTimeout(() => setShowIntro(true), introDelayMs);
    return () => window.clearTimeout(t);
  }, [introDelayMs]);

  function handleQuestion(qid: string, text: string, reply: string) {
    appendExchange({ questionId: qid, text, reply, source: 'scripted' });
    openFor(signal);
  }

  function handleOpenPanel() {
    openFor(signal);
  }

  return (
    <Card
      aria-label="Aperçu de la conversation avec Sentinel"
      className="relative flex flex-col gap-4 border-border/60 bg-card/80 p-5 shadow-md backdrop-blur sm:p-6"
    >
      <div className="flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <Bot className="h-4 w-4" aria-hidden />
        </div>
        <div>
          <p className="text-sm font-semibold">Sentinel</p>
          <p className="text-[11px] uppercase tracking-wider text-muted-foreground">
            Assistant conversationnel · {signal.instrument}
          </p>
        </div>
      </div>

      <div className="min-h-[120px]">
        {!showIntro ? (
          <ThinkingBubble />
        ) : (
          <div className="hero-stagger rounded-2xl rounded-tl-sm bg-muted px-4 py-3 text-sm leading-relaxed">
            <p>
              Salut, je suis Sentinel — l&apos;assistant de MIA Markets. Je
              viens de lire le marché {signal.instrument} : la structure
              indique une configuration{' '}
              {signal.direction === 'BULLISH_SETUP'
                ? 'haussière'
                : signal.direction === 'BEARISH_SETUP'
                  ? 'baissière'
                  : 'neutre'}
              .
            </p>
            <p className="mt-2 text-xs italic text-muted-foreground">
              Pose-moi n&apos;importe quelle question — je ne donne ni
              instruction d&apos;achat ni conseil personnalisé.
            </p>
          </div>
        )}
      </div>

      {showIntro && (
        <div
          className="hero-stagger flex flex-col gap-2"
          style={{ animationDelay: '300ms' }}
        >
          <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Questions suggérées
          </p>
          {previewQuestions.map((q) => (
            <button
              key={q.id}
              type="button"
              onClick={() => handleQuestion(q.id, q.text, q.reply)}
              className={cn(
                'rounded-lg border border-border bg-background px-3 py-2 text-left text-sm',
                'transition-colors hover:bg-accent hover:text-accent-foreground',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
              )}
            >
              {q.text}
            </button>
          ))}
        </div>
      )}

      {showIntro && (
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={handleOpenPanel}
          className="hero-stagger w-full"
          style={{ animationDelay: '500ms' }}
        >
          <MessageCircle className="h-4 w-4" aria-hidden />
          Continuer dans le chat
        </Button>
      )}
    </Card>
  );
}

function ThinkingBubble() {
  return (
    <div
      className="inline-flex items-center gap-1 rounded-2xl rounded-tl-sm bg-muted px-4 py-3 text-sm text-muted-foreground"
      aria-label="Sentinel réfléchit"
    >
      <span className="text-xs italic">Sentinel lit le marché</span>
      <span className="flex gap-1">
        <span
          className="hero-thinking-dot h-1.5 w-1.5 rounded-full bg-muted-foreground"
          style={{ animationDelay: '0ms' }}
        />
        <span
          className="hero-thinking-dot h-1.5 w-1.5 rounded-full bg-muted-foreground"
          style={{ animationDelay: '160ms' }}
        />
        <span
          className="hero-thinking-dot h-1.5 w-1.5 rounded-full bg-muted-foreground"
          style={{ animationDelay: '320ms' }}
        />
      </span>
    </div>
  );
}
