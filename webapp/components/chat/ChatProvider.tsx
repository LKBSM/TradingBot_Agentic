'use client';

import * as React from 'react';
import type { InsightSignalV2 } from '@/types/insight';

interface ChatTurn {
  id: string;
  role: 'user' | 'assistant';
  text: string;
}

interface ChatContextValue {
  /** True while the slide-over is mounted and visible. */
  isOpen: boolean;
  /** The signal whose context drives the current conversation. */
  activeSignal: InsightSignalV2 | null;
  /** Turns of the current conversation (cleared on signal switch). */
  turns: ChatTurn[];
  openFor(signal: InsightSignalV2): void;
  close(): void;
  /** Append a user question + the scripted assistant reply atomically. */
  appendExchange(args: { questionId: string; text: string; reply: string }): void;
  /** Reset the turn list (e.g. when the user re-opens the panel). */
  resetTurns(): void;
}

const ChatContext = React.createContext<ChatContextValue | null>(null);

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [isOpen, setIsOpen] = React.useState(false);
  const [activeSignal, setActiveSignal] = React.useState<InsightSignalV2 | null>(
    null,
  );
  const [turns, setTurns] = React.useState<ChatTurn[]>([]);
  const seqRef = React.useRef(0);

  const openFor = React.useCallback((signal: InsightSignalV2) => {
    setActiveSignal((current) => {
      // Switching signals clears the turn history to avoid mixing context.
      if (current?.id !== signal.id) {
        setTurns([]);
        seqRef.current = 0;
      }
      return signal;
    });
    setIsOpen(true);
  }, []);

  const close = React.useCallback(() => setIsOpen(false), []);

  const appendExchange = React.useCallback(
    ({ questionId, text, reply }: { questionId: string; text: string; reply: string }) => {
      setTurns((prev) => [
        ...prev,
        { id: `${questionId}-q-${seqRef.current++}`, role: 'user', text },
        { id: `${questionId}-a-${seqRef.current++}`, role: 'assistant', text: reply },
      ]);
    },
    [],
  );

  const resetTurns = React.useCallback(() => {
    setTurns([]);
    seqRef.current = 0;
  }, []);

  const value = React.useMemo<ChatContextValue>(
    () => ({
      isOpen,
      activeSignal,
      turns,
      openFor,
      close,
      appendExchange,
      resetTurns,
    }),
    [isOpen, activeSignal, turns, openFor, close, appendExchange, resetTurns],
  );

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat(): ChatContextValue {
  const ctx = React.useContext(ChatContext);
  if (!ctx) {
    throw new Error(
      'useChat must be used inside a <ChatProvider /> (mounted in [locale]/layout.tsx).',
    );
  }
  return ctx;
}
