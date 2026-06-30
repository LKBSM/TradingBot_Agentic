'use client';

import { HelpCircle, LayoutPanelTop, LineChart, RotateCcw } from 'lucide-react';
import { AgentAvatar } from '@/components/chat/AgentAvatar';
import { ChatInput } from '@/components/chat/ChatInput';
import { ChatMessage } from '@/components/chat/ChatMessage';
import { ChatWelcome, type WelcomeSuggestion } from '@/components/chat/ChatWelcome';
import { ThinkingIndicator } from '@/components/chat/ThinkingIndicator';
import { useChat } from '@/components/chat/ChatProvider';
import { useChatAnchorScroll } from '@/components/chat/useChatAnchorScroll';
import { Button } from '@/components/ui/button';
import {
  formatInstrument,
  formatTimeframe,
} from '@/lib/market-reading/formatters';
import type { Combo } from '@/lib/market-reading/store';

/** On-brand starter questions for the empty state — go through the live LLM. */
const STARTERS: ReadonlyArray<WelcomeSuggestion> = [
  {
    id: 'structure',
    text: 'Décompose la structure actuelle',
    icon: <LineChart className="h-4 w-4" aria-hidden />,
  },
  {
    id: 'choch',
    text: 'C’est quoi un CHOCH ?',
    icon: <HelpCircle className="h-4 w-4" aria-hidden />,
  },
  {
    id: 'order-blocks',
    text: 'Montre-moi les Order Blocks actifs',
    icon: <LayoutPanelTop className="h-4 w-4" aria-hidden />,
  },
];

/**
 * Right column — the permanently docked Sentinel chat. Unlike the layout's
 * slide-over Sheet (used on the landing), this renders inline and is always
 * visible on /app. It shares the same ChatProvider context; the active combo's
 * context is bound via `openForCombo` upstream (no Sheet, no modal).
 */
export function AppChatSidebar({ active }: { active: Combo | null }) {
  const { turns, isLoading, apiAvailable, askFreeForm, resetTurns } = useChat();
  // Keep main's anchor-scroll UX (anchor the latest question near the top after
  // sending instead of jumping to the bottom of a long reply) in the new sidebar.
  const scrollRef = useChatAnchorScroll(turns, isLoading);

  const empty = turns.length === 0;
  const offline = apiAvailable === false;

  function handleStarter(s: WelcomeSuggestion) {
    if (!active || offline) return;
    void askFreeForm(s.text);
  }

  return (
    <aside
      aria-label="Assistant M.I.A Agent"
      className="flex h-full min-h-0 flex-col rounded-xl border border-border/60 bg-card"
    >
      <header className="flex items-center gap-3 border-b border-border/60 px-4 py-3">
        <AgentAvatar size="md" />
        <div className="min-w-0 flex-1">
          <p className="flex items-center gap-1.5 text-sm font-semibold leading-tight">
            M.I.A Agent
            <span
              className="h-1.5 w-1.5 rounded-full bg-[hsl(var(--sentinel-bull))]"
              title={offline ? 'mode hors-ligne' : 'en ligne'}
              aria-hidden
            />
          </p>
          <p className="truncate text-xs text-muted-foreground">
            {active
              ? `${formatInstrument(active.instrument)} · ${formatTimeframe(active.timeframe)}`
              : 'Sélectionnez une combinaison pour discuter de sa lecture.'}
          </p>
          <p className="mt-0.5 text-[10.5px] italic text-muted-foreground/85">
            Analyse pédagogique — aucun signal ni conseil.
          </p>
        </div>
        {!empty && (
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={resetTurns}
            className="h-7 shrink-0 gap-1 px-2 text-xs text-muted-foreground"
          >
            <RotateCcw className="h-3 w-3" aria-hidden />
            <span className="sr-only sm:not-sr-only">Réinitialiser</span>
          </Button>
        )}
      </header>

      <div
        ref={scrollRef}
        className="flex flex-1 flex-col gap-4 overflow-y-auto px-4 py-4"
      >
        {empty ? (
          <ChatWelcome
            title={
              active
                ? 'Comment puis-je t’aider à lire le marché ?'
                : 'Choisis un marché pour commencer'
            }
            subtitle={
              active
                ? 'Pose une question sur la lecture en cours : décompose la structure, vulgarise un terme, ou contextualise un événement à venir.'
                : 'Sélectionne un marché à gauche, puis pose-moi une question sur sa lecture.'
            }
            suggestions={active && !offline ? STARTERS : []}
            onPick={handleStarter}
            note={
              offline
                ? 'Mode hors-ligne : la saisie libre nécessite la clef Anthropic côté serveur.'
                : undefined
            }
          />
        ) : (
          <>
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
          </>
        )}
      </div>

      <div className="space-y-2 border-t border-border/60 bg-background/60 px-4 py-3">
        <ChatInput />
        {/* LEGAL-PENDING: chat compliance line — aligned with the legal terminal
            wording on educational-use posture (UE 2024/2811 + MiFID II 03/2026). */}
        <p className="text-center text-[10.5px] italic text-muted-foreground/70">
          M.I.A Agent répond à des questions sur la lecture algorithmique. Il ne
          donne ni signal de trading, ni recommandation personnalisée.
        </p>
      </div>
    </aside>
  );
}
