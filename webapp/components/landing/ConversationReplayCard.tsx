'use client';

import { useTranslations } from 'next-intl';
import { Play, RotateCcw, ShieldAlert, User } from 'lucide-react';
import * as React from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { MiaAgentLogo } from '@/components/chat/MiaAgentLogo';
import { cn } from '@/lib/utils';

interface ConversationReplayCardProps {
  title: string;
  kicker: string;
  question: string;
  answer: string;
  instrument: string;
  highlight?: 'refusal' | 'normal';
}

type ReplayState = 'idle' | 'asking' | 'thinking' | 'answering' | 'done';

const ASK_DELAY = 800; // Time the question stays alone before Sentinel "thinks"
const THINK_DELAY = 900; // Thinking-dots duration
const TYPE_INTERVAL = 14; // ms per character during the typing effect

/**
 * One conversation tile that "plays" : user bubble appears, Sentinel
 * thinks (3 dots), then types out the scripted reply character by
 * character. The "Rejouer" button restarts the playback. The refusal
 * variant gets a subtle warn-tinted border to draw the eye to the moat
 * differentiator (chatbot says no to instructions).
 */
export function ConversationReplayCard({
  title,
  kicker,
  question,
  answer,
  instrument,
  highlight = 'normal',
}: ConversationReplayCardProps) {
  const t = useTranslations('landing.replayCard');
  const [state, setState] = React.useState<ReplayState>('idle');
  const [typedChars, setTypedChars] = React.useState(0);
  const cardRef = React.useRef<HTMLDivElement>(null);
  const startedRef = React.useRef(false);

  // Auto-play when the card scrolls into view (only once).
  React.useEffect(() => {
    if (!cardRef.current) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const first = entries[0];
        if (first?.isIntersecting && !startedRef.current) {
          startedRef.current = true;
          start();
        }
      },
      { threshold: 0.4 },
    );
    observer.observe(cardRef.current);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const start = React.useCallback(() => {
    setState('asking');
    setTypedChars(0);
    const t1 = window.setTimeout(() => setState('thinking'), ASK_DELAY);
    const t2 = window.setTimeout(
      () => setState('answering'),
      ASK_DELAY + THINK_DELAY,
    );
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
    };
  }, []);

  // Type the answer one char at a time while in 'answering' state.
  React.useEffect(() => {
    if (state !== 'answering') return;
    if (typedChars >= answer.length) {
      setState('done');
      return;
    }
    const id = window.setTimeout(
      () => setTypedChars((n) => n + 1),
      TYPE_INTERVAL,
    );
    return () => window.clearTimeout(id);
  }, [state, typedChars, answer.length]);

  function handleReplay() {
    setTypedChars(0);
    setState('idle');
    window.setTimeout(() => start(), 50);
  }

  const showQuestion = state !== 'idle';
  const showThinking = state === 'thinking';
  const showAnswer = state === 'answering' || state === 'done';
  const displayedAnswer = state === 'done' ? answer : answer.slice(0, typedChars);

  return (
    <Card
      ref={cardRef}
      className={cn(
        'flex h-full flex-col border-border/60 shadow-sm',
        highlight === 'refusal' && 'border-sentinel-warn/40 shadow-md',
      )}
    >
      <CardContent className="flex h-full flex-col gap-4 p-5 sm:p-6">
        <header className="flex items-start justify-between gap-2">
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              {kicker}
            </p>
            <h3 className="mt-0.5 text-base font-semibold tracking-tight">
              {title}
            </h3>
          </div>
          {highlight === 'refusal' && (
            <Badge variant="warn" className="shrink-0 text-[10px]">
              <ShieldAlert className="mr-1 h-3 w-3" aria-hidden />
              {t('compliance')}
            </Badge>
          )}
        </header>

        <Badge variant="outline" className="w-fit text-[10px]">
          {t('contextInjected')} · {instrument}
        </Badge>

        <div className="flex flex-1 flex-col gap-3 text-sm">
          {showQuestion && (
            <Bubble role="user">
              <User className="h-3.5 w-3.5" aria-hidden />
              {question}
            </Bubble>
          )}

          {showThinking && (
            <Bubble role="assistant">
              <MiaAgentLogo className="h-3.5 w-3.5 shrink-0" />
              <ThinkingDots />
            </Bubble>
          )}

          {showAnswer && (
            <Bubble role="assistant">
              <MiaAgentLogo className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span className="whitespace-pre-wrap leading-relaxed">
                {displayedAnswer}
                {state === 'answering' && <Caret />}
              </span>
            </Bubble>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 pt-2">
          {state === 'idle' ? (
            <Button type="button" size="sm" variant="ghost" onClick={start}>
              <Play className="h-3.5 w-3.5" aria-hidden />
              {t('play')}
            </Button>
          ) : (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={handleReplay}
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden />
              {t('replay')}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function Bubble({
  role,
  children,
}: {
  role: 'user' | 'assistant';
  children: React.ReactNode;
}) {
  const isUser = role === 'user';
  return (
    <div
      className={cn(
        'hero-stagger flex max-w-[90%] gap-2 rounded-2xl px-3 py-2 text-xs sm:text-sm',
        isUser
          ? 'self-end rounded-tr-sm bg-primary text-primary-foreground'
          : 'self-start rounded-tl-sm bg-muted text-foreground',
      )}
    >
      {children}
    </div>
  );
}

function ThinkingDots() {
  return (
    <span className="inline-flex items-center gap-1">
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
  );
}

function Caret() {
  return (
    <span
      aria-hidden
      className="ml-0.5 inline-block h-3 w-[2px] -translate-y-px animate-pulse bg-current align-middle"
    />
  );
}
