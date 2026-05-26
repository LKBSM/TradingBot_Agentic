'use client';

import * as React from 'react';
import {
  askSentinel,
  ChatApiUnavailableError,
} from '@/lib/chat/api-client';
import type { InsightSignalV2 } from '@/types/insight';

interface ChatTurn {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  /** Was this answer produced by the live LLM API or the scripted fallback? */
  source?: 'llm' | 'scripted' | 'fallback' | 'error';
}

interface ChatContextValue {
  isOpen: boolean;
  activeSignal: InsightSignalV2 | null;
  turns: ChatTurn[];
  /** True while a streaming LLM response is being generated. */
  isStreaming: boolean;
  /** Partial text streamed so far (cleared once finalised into `turns`). */
  streamingText: string;
  /**
   * Whether the /api/chat backend is wired (`true` = LLM live, `false` =
   * server returned 503, scripted fallback only, `'unknown'` before first
   * call). Sticky once set to false to avoid spamming the endpoint.
   */
  apiAvailable: boolean | 'unknown';
  openFor(signal: InsightSignalV2): void;
  close(): void;
  appendExchange(args: {
    questionId: string;
    text: string;
    reply: string;
    source?: ChatTurn['source'];
  }): void;
  /**
   * Submit a free-form user question. Calls /api/chat with streaming, falls
   * back to a friendly message if the API is unavailable. Throws if no
   * activeSignal is set.
   */
  askFreeForm(question: string): Promise<void>;
  resetTurns(): void;
}

const ChatContext = React.createContext<ChatContextValue | null>(null);

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [isOpen, setIsOpen] = React.useState(false);
  const [activeSignal, setActiveSignal] = React.useState<InsightSignalV2 | null>(
    null,
  );
  const [turns, setTurns] = React.useState<ChatTurn[]>([]);
  const [isStreaming, setIsStreaming] = React.useState(false);
  const [streamingText, setStreamingText] = React.useState('');
  const [apiAvailable, setApiAvailable] = React.useState<boolean | 'unknown'>(
    'unknown',
  );
  const seqRef = React.useRef(0);

  const openFor = React.useCallback((signal: InsightSignalV2) => {
    setActiveSignal((current) => {
      if (current?.id !== signal.id) {
        setTurns([]);
        setStreamingText('');
        seqRef.current = 0;
      }
      return signal;
    });
    setIsOpen(true);
  }, []);

  const close = React.useCallback(() => setIsOpen(false), []);

  const nextId = React.useCallback((prefix: string) => {
    return `${prefix}-${seqRef.current++}`;
  }, []);

  const appendExchange = React.useCallback(
    ({
      questionId,
      text,
      reply,
      source = 'scripted',
    }: {
      questionId: string;
      text: string;
      reply: string;
      source?: ChatTurn['source'];
    }) => {
      setTurns((prev) => [
        ...prev,
        { id: `${questionId}-q-${seqRef.current++}`, role: 'user', text },
        {
          id: `${questionId}-a-${seqRef.current++}`,
          role: 'assistant',
          text: reply,
          source,
        },
      ]);
    },
    [],
  );

  const resetTurns = React.useCallback(() => {
    setTurns([]);
    setStreamingText('');
    seqRef.current = 0;
  }, []);

  const askFreeForm = React.useCallback(
    async (question: string) => {
      if (!activeSignal) {
        throw new Error('No active signal for chat');
      }
      const trimmed = question.trim();
      if (!trimmed) return;

      // Push the user turn immediately so the UI feels responsive.
      const userTurnId = nextId('user');
      setTurns((prev) => [
        ...prev,
        { id: userTurnId, role: 'user', text: trimmed },
      ]);

      // Build history payload (last 6 turns, alternating).
      const historyForApi = turns.slice(-6).map((t) => ({
        role: t.role,
        content: t.text,
      }));

      setIsStreaming(true);
      setStreamingText('');

      try {
        let accumulated = '';
        const { text } = await askSentinel({
          signal: activeSignal,
          question: trimmed,
          history: historyForApi,
          onDelta: (chunk) => {
            accumulated += chunk;
            setStreamingText(accumulated);
          },
        });
        setApiAvailable(true);
        // Convert streaming text into a final turn.
        setTurns((prev) => [
          ...prev,
          { id: nextId('asst'), role: 'assistant', text, source: 'llm' },
        ]);
      } catch (err) {
        if (err instanceof ChatApiUnavailableError) {
          setApiAvailable(false);
          setTurns((prev) => [
            ...prev,
            {
              id: nextId('asst'),
              role: 'assistant',
              source: 'fallback',
              text:
                "Le mode chatbot en direct n'est pas encore activé sur cet environnement. " +
                "Utilise les questions suggérées ci-dessous — elles renvoient des réponses pré-écrites contextualisées sur cette lecture.",
            },
          ]);
        } else {
          const message = err instanceof Error ? err.message : 'Erreur inconnue';
          setTurns((prev) => [
            ...prev,
            {
              id: nextId('asst'),
              role: 'assistant',
              source: 'error',
              text: `Désolé, une erreur a empêché la réponse : ${message}. Réessaie ou utilise une question suggérée.`,
            },
          ]);
        }
      } finally {
        setStreamingText('');
        setIsStreaming(false);
      }
    },
    [activeSignal, turns, nextId],
  );

  const value = React.useMemo<ChatContextValue>(
    () => ({
      isOpen,
      activeSignal,
      turns,
      isStreaming,
      streamingText,
      apiAvailable,
      openFor,
      close,
      appendExchange,
      askFreeForm,
      resetTurns,
    }),
    [
      isOpen,
      activeSignal,
      turns,
      isStreaming,
      streamingText,
      apiAvailable,
      openFor,
      close,
      appendExchange,
      askFreeForm,
      resetTurns,
    ],
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
