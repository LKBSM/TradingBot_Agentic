'use client';

import {
  HelpCircle,
  History,
  LayoutPanelTop,
  LineChart,
  RotateCcw,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import * as React from 'react';
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
  formatRelativePast,
  formatTimeframe,
} from '@/lib/market-reading/formatters';
import type { Combo } from '@/lib/market-reading/store';

/** Icons for the on-brand starter questions (text is localized in-component). */
const STARTER_META: ReadonlyArray<{ id: string; icon: React.ReactNode }> = [
  { id: 'structure', icon: <LineChart className="h-4 w-4" aria-hidden /> },
  { id: 'choch', icon: <HelpCircle className="h-4 w-4" aria-hidden /> },
  { id: 'order-blocks', icon: <LayoutPanelTop className="h-4 w-4" aria-hidden /> },
];

/**
 * Right column — the permanently docked Sentinel chat. Unlike the layout's
 * slide-over Sheet (used on the landing), this renders inline and is always
 * visible on /app. It shares the same ChatProvider context; the active combo's
 * context is bound via `openForCombo` upstream (no Sheet, no modal).
 */
export function AppChatSidebar({
  active,
  onSelectCombo,
}: {
  active: Combo | null;
  /**
   * Switch the workspace to another combo (wired to the same `onSelect` as the
   * instruments column) — used by the recent-discussions list to jump back to
   * a combo's conversation. Optional so the sidebar renders standalone.
   */
  onSelectCombo?: (combo: Combo) => void;
}) {
  const t = useTranslations('app');
  const {
    turns,
    isLoading,
    apiAvailable,
    askFreeForm,
    resetTurns,
    recentThreads,
  } = useChat();
  const STARTERS: ReadonlyArray<WelcomeSuggestion> = STARTER_META.map((s) => ({
    id: s.id,
    text: t(`chat.starter_${s.id}`),
    icon: s.icon,
  }));
  const [showRecents, setShowRecents] = React.useState(false);
  // Keep main's anchor-scroll UX (anchor the latest question near the top after
  // sending instead of jumping to the bottom of a long reply) in the new sidebar.
  const scrollRef = useChatAnchorScroll(turns, isLoading);

  const empty = turns.length === 0;
  const offline = apiAvailable === false;
  const activeThreadId = active
    ? `app:${active.instrument}:${active.timeframe}`
    : null;

  function handleStarter(s: WelcomeSuggestion) {
    if (!active || offline) return;
    void askFreeForm(s.text);
  }

  return (
    <aside
      aria-label={t('chat.asideAria')}
      className="flex h-full min-h-0 flex-col rounded-xl border border-border/60 bg-card"
    >
      <header className="flex items-center gap-3 border-b border-border/60 px-4 py-3">
        <AgentAvatar size="md" />
        <div className="min-w-0 flex-1">
          <p className="flex items-center gap-1.5 text-sm font-semibold leading-tight">
            M.I.A Agent
            <span
              className="h-1.5 w-1.5 rounded-full bg-[hsl(var(--sentinel-bull))]"
              title={offline ? t('chat.statusOffline') : t('chat.statusOnline')}
              aria-hidden
            />
          </p>
          <p className="truncate text-xs text-muted-foreground">
            {active
              ? `${formatInstrument(active.instrument)} · ${formatTimeframe(active.timeframe)}`
              : t('chat.pickComboPrompt')}
          </p>
          <p className="mt-0.5 text-[10.5px] italic text-muted-foreground/85">
            {t('chat.pedagogicalNote')}
          </p>
        </div>
        {recentThreads.length > 0 && (
          <Button
            type="button"
            size="sm"
            variant="ghost"
            aria-expanded={showRecents}
            onClick={() => setShowRecents((v) => !v)}
            className="h-7 shrink-0 gap-1 px-2 text-xs text-muted-foreground"
          >
            <History className="h-3 w-3" aria-hidden />
            <span className="sr-only sm:not-sr-only">{t('chat.discussions')}</span>
          </Button>
        )}
        {!empty && (
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={resetTurns}
            className="h-7 shrink-0 gap-1 px-2 text-xs text-muted-foreground"
          >
            <RotateCcw className="h-3 w-3" aria-hidden />
            <span className="sr-only sm:not-sr-only">{t('chat.reset')}</span>
          </Button>
        )}
      </header>

      {showRecents && recentThreads.length > 0 && (
        <nav
          aria-label={t('chat.recentDiscussions')}
          className="border-b border-border/60 px-2 py-2"
        >
          <p className="px-2 pb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            {t('chat.recentDiscussions')}
          </p>
          <ul className="space-y-0.5">
            {recentThreads.map((t) => {
              const isActive = t.id === activeThreadId;
              return (
                <li key={t.id}>
                  <button
                    type="button"
                    onClick={() => {
                      if (!isActive) {
                        onSelectCombo?.({
                          instrument: t.instrument,
                          timeframe: t.timeframe,
                        });
                      }
                      setShowRecents(false);
                    }}
                    className={`w-full rounded-md px-2 py-1.5 text-left transition-colors hover:bg-muted ${
                      isActive ? 'bg-muted/70' : ''
                    }`}
                  >
                    <span className="flex items-baseline justify-between gap-2">
                      <span className="text-xs font-medium">
                        {formatInstrument(t.instrument)} ·{' '}
                        {formatTimeframe(t.timeframe)}
                      </span>
                      <span className="shrink-0 text-[10.5px] text-muted-foreground">
                        {formatRelativePast(new Date(t.updatedAt).toISOString())}
                      </span>
                    </span>
                    {t.lastText && (
                      <span className="block truncate text-[11px] text-muted-foreground">
                        {t.lastText}
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>
      )}

      <div
        ref={scrollRef}
        className="flex flex-1 flex-col gap-4 overflow-y-auto px-4 py-4"
      >
        {empty ? (
          <ChatWelcome
            title={active ? t('chat.welcomeTitleActive') : t('chat.welcomeTitleIdle')}
            subtitle={
              active ? t('chat.welcomeSubtitleActive') : t('chat.welcomeSubtitleIdle')
            }
            suggestions={active && !offline ? STARTERS : []}
            onPick={handleStarter}
            note={offline ? t('chat.offlineNote') : undefined}
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
            wording on educational-use posture. */}
        <p className="text-center text-[10.5px] italic text-muted-foreground/70">
          {t('chat.complianceLine')}
        </p>
      </div>
    </aside>
  );
}
