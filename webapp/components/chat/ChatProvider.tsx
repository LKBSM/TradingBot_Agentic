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
  /**
   * Set when a niveau-1.5 defence layer redirected the answer (adversarial
   * category, `llm_error`, `output_contaminated_*`, …). `text` already carries
   * the pedagogical template; the UI shows a discreet badge. Null on a normal
   * answer.
   */
  blockedReason?: string | null;
}

interface ChatContextValue {
  isOpen: boolean;
  activeSignal: InsightSignalV2 | null;
  turns: ChatTurn[];
  /** True while a backend answer is in flight (synchronous JSON, no stream). */
  isLoading: boolean;
  /**
   * Whether the backend chatbot is reachable (`true` = answering, `false` =
   * endpoint returned 503 / not bootstrapped, scripted fallback only,
   * `'unknown'` before the first call). Sticky once set to false to avoid
   * spamming the endpoint.
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
   * Submit a free-form user question. Calls POST /api/chatbot/message (3 niveau-
   * 1.5 defence layers server-side), falls back to a friendly message if the
   * backend is unavailable. Throws if no activeSignal is set.
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
  const [isLoading, setIsLoading] = React.useState(false);
  const [apiAvailable, setApiAvailable] = React.useState<boolean | 'unknown'>(
    'unknown',
  );
  const seqRef = React.useRef(0);

  const openFor = React.useCallback((signal: InsightSignalV2) => {
    setActiveSignal((current) => {
      if (current?.id !== signal.id) {
        setTurns([]);
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

      // Build history payload (last 6 turns, alternating user/assistant).
      const historyForApi = turns.slice(-6).map((t) => ({
        role: t.role,
        content: t.text,
      }));

      setIsLoading(true);

      try {
        const { text, blockedReason } = await askSentinel({
          signal: activeSignal,
          question: trimmed,
          history: historyForApi,
        });
        setApiAvailable(true);
        setTurns((prev) => [
          ...prev,
          {
            id: nextId('asst'),
            role: 'assistant',
            text,
            source: 'llm',
            blockedReason,
          },
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
                "Le mode chatbot en direct n'est pas disponible sur cet environnement pour le moment. " +
                'Utilise les questions suggérées ci-dessous — elles renvoient des réponses contextualisées sur cette lecture.',
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
              text: `Désolé, une erreur a empêché la réponse : ${message} Réessaie ou utilise une question suggérée.`,
            },
          ]);
        }
      } finally {
        setIsLoading(false);
      }
    },
    [activeSignal, turns, nextId],
  );

  const value = React.useMemo<ChatContextValue>(
    () => ({
      isOpen,
      activeSignal,
      turns,
      isLoading,
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
      isLoading,
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
