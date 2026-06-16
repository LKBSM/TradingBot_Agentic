'use client';

import { MessageCircle } from 'lucide-react';
import * as React from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { MiaAgentLogo } from '@/components/chat/MiaAgentLogo';
import { useChat } from '@/components/chat/ChatProvider';
import { getChatbotScript } from '@/lib/chatbot';
import { cn } from '@/lib/utils';
import type { LandingSample } from '@/lib/market-reading/landing-samples';

interface HeroChatPreviewProps {
  sample: LandingSample;
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
  sample,
  introDelayMs = 1400,
}: HeroChatPreviewProps) {
  const { openFor, appendExchange } = useChat();
  const [showIntro, setShowIntro] = React.useState(false);

  const { reading } = sample;
  const chatContext = React.useMemo(
    () => ({
      id: sample.id,
      instrument: reading.header.instrument,
      timeframe: reading.header.timeframe,
    }),
    [sample.id, reading.header.instrument, reading.header.timeframe],
  );

  const script = React.useMemo(() => getChatbotScript(sample.id), [sample.id]);
  const previewQuestions = React.useMemo(
    () => script?.questions.slice(0, 3) ?? [],
    [script],
  );

  React.useEffect(() => {
    // Delay the intro bubble so the card finishes its composition first,
    // creating the impression that Sentinel "just finished reading" the
    // reading before greeting.
    const t = window.setTimeout(() => setShowIntro(true), introDelayMs);
    return () => window.clearTimeout(t);
  }, [introDelayMs]);

  function handleQuestion(qid: string, text: string, reply: string) {
    appendExchange({ questionId: qid, text, reply, source: 'scripted' });
    openFor(chatContext);
  }

  function handleOpenPanel() {
    openFor(chatContext);
  }

  return (
    <Card
      aria-label="Aperçu de la conversation avec M.I.A Agent"
      className="relative flex flex-col gap-4 border-border/60 bg-card/80 p-5 shadow-md backdrop-blur sm:p-6"
    >
      <div className="flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <MiaAgentLogo className="h-5 w-5" />
        </div>
        <div>
          <p className="text-sm font-semibold">M.I.A Agent</p>
          <p className="text-[11px] uppercase tracking-wider text-muted-foreground">
            Assistant conversationnel · {reading.header.instrument}
          </p>
        </div>
      </div>

      <div className="min-h-[120px]">
        {!showIntro ? (
          <ThinkingBubble />
        ) : (
          <div className="hero-stagger rounded-2xl rounded-tl-sm bg-muted px-4 py-3 text-sm leading-relaxed">
            <p>
              Salut, je suis M.I.A Agent — l&apos;assistant de MIA Markets. Je
              viens de lire le marché {reading.header.instrument} : la structure
              indique une configuration{' '}
              {reading.regime.trend === 'bullish'
                ? 'haussière'
                : reading.regime.trend === 'bearish'
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
      aria-label="M.I.A Agent réfléchit"
    >
      <span className="text-xs italic">M.I.A Agent lit le marché</span>
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
