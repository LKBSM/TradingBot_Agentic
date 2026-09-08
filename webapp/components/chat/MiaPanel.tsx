'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';
import { AgentAvatar } from '@/components/chat/AgentAvatar';
import { ChatInput } from '@/components/chat/ChatInput';
import { ChatMessage } from '@/components/chat/ChatMessage';
import { ChatWelcome, type WelcomeSuggestion } from '@/components/chat/ChatWelcome';
import { ThinkingIndicator } from '@/components/chat/ThinkingIndicator';
import { useChat } from '@/components/chat/ChatProvider';
import { useChatAnchorScroll } from '@/components/chat/useChatAnchorScroll';

/**
 * MIA-3 — THE single M.I.A conversation panel, used on every product surface
 * (docked column / bubble drawer on /app via AppChatSidebar, the /zones aside,
 * the /actualites publication block). There is exactly ONE conversation-panel
 * implementation in the code: the transcript, the welcome, the thinking
 * indicator, the input and the compliance line all live here and read the single
 * shared conversation from `useChat()`. Surfaces differ only by the chrome they
 * pass in (a context label, an orientation subject, header actions) — never by a
 * second copy of the conversation UI.
 *
 * It NEVER carries its own answering logic: every fact comes from the backend
 * agent's tools. The `subject` block is pure orientation and is simply NOT
 * rendered when absent (no empty box, no filler) — clearing a selection removes
 * the block without touching the conversation.
 */
export interface MiaPanelProps {
  /** Accessible name for the panel region. */
  ariaLabel: string;
  /** Heading shown next to the avatar (e.g. "M.I.A" / "M.I.A Agent"). */
  title: string;
  /** Live context after the title (e.g. "· XAUUSD · M15"), optional. */
  contextLabel?: React.ReactNode;
  /** Right-aligned header actions (e.g. the /app display-mode toggle). */
  headerActions?: React.ReactNode;
  /** A status line under the header (e.g. the /app mode-preference note). */
  statusLine?: React.ReactNode;
  /** Empty-state welcome heading. */
  welcomeTitle: string;
  /** Empty-state welcome subheading. */
  welcomeSubtitle: string;
  /** Descriptive starter questions (empty state); omitted when offline. */
  starters?: ReadonlyArray<WelcomeSuggestion>;
  /** Offline note in the empty state. */
  offlineNote?: string;
  /**
   * Per-panel compliance line under the input. Pass ONLY where the panel is the
   * page's disclaimer carrier (/app). Omitted on pages that already show the
   * single page disclaimer elsewhere (/zones, /actualites) — one per page.
   */
  complianceLine?: string;
  className?: string;
}

export function MiaPanel({
  ariaLabel,
  title,
  contextLabel,
  headerActions,
  statusLine,
  welcomeTitle,
  welcomeSubtitle,
  starters,
  offlineNote,
  complianceLine,
  className,
}: MiaPanelProps) {
  const { turns, isLoading, apiAvailable, askFreeForm, activeSignal, focus } =
    useChat();
  // Orientation subject — driven by the SHARED focus (selected zone / open
  // publication), so it shows in the one panel wherever it is docked. Rendered
  // ONLY when a focus is set: no zone / no publication → no block at all, never
  // an empty box or filler (mission §3/§4). `label` is display-only.
  const subject = focus ? (
    <div
      className="flex items-center gap-1.5 border-b border-border/60 px-4 py-2 text-xs font-medium"
      data-testid="mia-subject"
    >
      <span
        className="h-1.5 w-1.5 shrink-0 rounded-full bg-[hsl(var(--sentinel-accent,var(--primary)))]"
        aria-hidden
      />
      <span className="truncate">{focus.label}</span>
    </div>
  ) : null;
  const empty = turns.length === 0;
  const offline = apiAvailable === false;
  // Anchor the first word of M.I.A's reply to the top after sending (MIA-1).
  const scrollRef = useChatAnchorScroll(turns, isLoading, { anchor: 'assistant' });

  const handleStarter = React.useCallback(
    (s: WelcomeSuggestion) => {
      if (!activeSignal || offline) return;
      void askFreeForm(s.text);
    },
    [activeSignal, offline, askFreeForm],
  );

  return (
    <aside
      aria-label={ariaLabel}
      className={cn(
        'flex h-full min-h-0 flex-col rounded-xl border border-border/60 bg-card',
        className,
      )}
    >
      <header className="border-b border-border/60 px-4 py-3">
        <div className="flex items-center gap-3">
          <AgentAvatar size="md" presence />
          <p className="flex min-w-0 flex-1 items-center gap-1.5 text-sm font-semibold leading-tight">
            <span className="shrink-0">{title}</span>
            {contextLabel != null && (
              <span className="truncate text-xs font-normal text-muted-foreground">
                {contextLabel}
              </span>
            )}
          </p>
          {headerActions != null && (
            <div className="flex shrink-0 items-center gap-0.5">{headerActions}</div>
          )}
        </div>
        {statusLine != null && statusLine}
      </header>

      {/* Orientation subject — only when the surface provides one. */}
      {subject != null && subject}

      <div
        ref={scrollRef}
        className="flex flex-1 flex-col gap-4 overflow-y-auto px-4 py-4"
        role="log"
        aria-live="polite"
        aria-relevant="additions"
      >
        {empty ? (
          <ChatWelcome
            title={welcomeTitle}
            subtitle={welcomeSubtitle}
            suggestions={!offline ? (starters ?? []) : []}
            onPick={handleStarter}
            note={offline ? offlineNote : undefined}
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
        {complianceLine != null && (
          <p className="text-center text-[11px] italic text-muted-foreground/70">
            {complianceLine}
          </p>
        )}
      </div>
    </aside>
  );
}
